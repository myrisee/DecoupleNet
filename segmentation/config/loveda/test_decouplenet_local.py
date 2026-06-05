"""
LoveDA Local Test Config — Windows/Linux local icin.
Colab bagimliliklari yok; kendi bilgisayarinda calisir.

Kullanim:
  Validation:  python loveda_test.py -c config/loveda/test_decouplenet_local.py -o fig_results/loveda/local_val -t 'd4' --val
  Test:        python loveda_test.py -c config/loveda/test_decouplenet_local.py -o fig_results/loveda/local_test -t 'd4'

Beklenen dizin yapisi (data_root altinda):
  Train/
    Urban/images_png/       (egitim icin, testte gerekmez)
    Urban/masks_png/
    Rural/images_png/
    Rural/masks_png/
  Val/
    Urban/images_png/
    Urban/masks_png/
    Rural/images_png/
    Rural/masks_png/
  Test/
    Urban/images_png/
    Rural/images_png/
"""

from torch.utils.data import DataLoader
from geoseg.losses import UnetFormerLoss
from geoseg.datasets.loveda_dataset import LoveDATrainDataset, LoveDATestDataset, CLASSES
from geoseg.models.UNetFormer_decouplenet import UNetFormer_DecoupleNet_D2
from geoseg.datasets.transform import Compose
import torch
import numpy as np
import albumentations as albu

# ===================== PARAMETRELER =====================
max_epoch = 30
ignore_index = len(CLASSES)
num_classes = len(CLASSES)
classes = CLASSES

# Agirlik ve log ayarlari — KENDI YOLUNU GUNCELLE
data_root = "data/LoveDA"                              # LoveDA kok dizini
weights_path = "model_weights/loveda"                  # ckpt dosyasinin oldugu klasor
test_weights_name = "decouplenet-loveda-colab-epoch30" # .ckpt uzantisi olmadan
log_name = 'loveda/local-test'
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = 1

# ===================== MODEL =====================
net = UNetFormer_DecoupleNet_D2(num_classes=num_classes)

# ===================== LOSS (load_from_checkpoint icin gerekli) =====================
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

# ===================== TRANSFORMS =====================
def get_val_transform():
    val_transform = [albu.Normalize()]
    return albu.Compose(val_transform)

def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

# ===================== VERI SETLERI =====================
val_dataset = LoveDATrainDataset(
    data_root=f'{data_root}/Val',
    mosaic_ratio=0.0,
    transform=val_aug
)

test_dataset = LoveDATestDataset(
    data_root=f'{data_root}/Test'
)

train_loader = DataLoader(dataset=val_dataset, batch_size=2, num_workers=0, shuffle=False, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=2, num_workers=0, shuffle=False, pin_memory=True)

# ===================== OPTIMIZER (load_from_checkpoint icin gerekli) =====================
optimizer = torch.optim.AdamW(net.parameters(), lr=4e-4, weight_decay=0.01)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
