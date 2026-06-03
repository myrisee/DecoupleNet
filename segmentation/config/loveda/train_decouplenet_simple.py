"""
LoveDA ile DecoupleNet Eğitim Config - Basitleştirilmiş
Kullanım: python train_supervision.py -c config/loveda/train_decouplenet_simple.py
"""

from torch.utils.data import DataLoader
from geoseg.losses import UnetFormerLoss
from geoseg.datasets.loveda_dataset import LoveDATrainDataset, loveda_val_dataset, CLASSES
from geoseg.models.UNetFormer_decouplenet import UNetFormer_DecoupleNet_D2
from catalyst.contrib.nn import Lookahead
from catalyst import utils
import torch
import numpy as np
import albumentations as albu
from geoseg.datasets.transform import Compose, RandomScale, SmartCropV1

# ===================== EĞİTİM PARAMETRELERİ =====================
max_epoch = 30                      # Epoch sayısı (azaltılabilir: 20, 10)
ignore_index = len(CLASSES)         # 7 (background dahil toplam)
train_batch_size = 8               # Batch size (GPU belleğine göre: 16, 8, 4)
val_batch_size = 8
lr = 4e-4                          # Learning rate
weight_decay = 0.01
backbone_lr = 4e-4
backbone_weight_decay = 0.01
num_classes = len(CLASSES)          # 7 sınıf
classes = CLASSES                  # ('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural')

# Ağırlık ve log ayarları
weights_name = "decouplenet-loveda-epoch30"
weights_path = "model_weights/loveda"
test_weights_name = "last"         # veya "best-mIoU"
log_name = 'loveda/decouplenet'
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None        # Devam etmek için checkpoint path'i
resume_ckpt_path = None
gpus = 'auto'                      # 'auto', 1, 2, veya [0, 1]

# ===================== MODEL =====================
net = UNetFormer_DecoupleNet_D2(num_classes=num_classes)

# ===================== LOSS =====================
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

# ===================== DATA TRANSFORMS =====================
def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    # Multi-scale training ve crop
    crop_aug = Compose([
        RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)
    ])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


# ===================== VERİ SETLERİ =====================
# Windows path'lerini kullanın (ters slash dikkat)
train_dataset = LoveDATrainDataset(
    transform=train_aug,
    data_root='C:/Github/DecoupleNet/segmentation/data/LoveDA/Train',
    mosaic_ratio=0.25
)

val_dataset = LoveDATrainDataset(
    transform=val_aug,
    data_root='C:/Github/DecoupleNet/segmentation/data/LoveDA/Val',
    mosaic_ratio=0.0
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

# ===================== OPTİMİZER =====================
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
