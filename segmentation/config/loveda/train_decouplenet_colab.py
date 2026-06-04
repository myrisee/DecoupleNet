"""
LoveDA ile DecoupleNet Eğitim Config — Google Colab
T4 (16GB) / V100 (16GB) / A100 (40GB) icin optimize edilmis parametreler.

Kullanim: python train_supervision.py -c config/loveda/train_decouplenet_colab.py
"""

from torch.utils.data import DataLoader
from geoseg.losses import UnetFormerLoss
from geoseg.datasets.loveda_dataset_colab import LoveDAFlatTrainDataset, LoveDAFlatTestDataset, CLASSES
from geoseg.models.UNetFormer_decouplenet import UNetFormer_DecoupleNet_D2
import torch
import numpy as np
import albumentations as albu
from geoseg.datasets.transform import Compose, RandomScale, SmartCropV1


# ===================== LOOKAHEAD (catalyst yerine inline) =====================
class Lookahead(torch.optim.Optimizer):
    """Lookahead optimizer wrapper - catalyst dependency removed for Python 3.12+."""
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self.fast_step = 0
        self._copy_params()

    def _copy_params(self):
        self.slow_weights = [[p.clone().detach() for p in group['params']] for group in self.param_groups]

    def step(self, closure=None):
        result = self.optimizer.step(closure)
        self.fast_step += 1
        if self.fast_step >= self.k:
            self.fast_step = 0
            for group, slow_ws in zip(self.param_groups, self.slow_weights):
                for fast, slow in zip(group['params'], slow_ws):
                    if fast.grad is not None:
                        slow.add_(fast.data - slow, alpha=self.alpha)
                        fast.data.copy_(slow)
        return result

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {'fast_state': self.optimizer.state_dict(), 'slow_weights': self.slow_weights, 'fast_step': self.fast_step}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['fast_state'])
        self.slow_weights = state_dict['slow_weights']
        self.fast_step = state_dict['fast_step']


def process_model_params(model, layerwise_params=None):
    """Group model parameters with optional per-layer lr/weight_decay."""
    if layerwise_params is None:
        return [dict(params=list(model.parameters()))]
    param_groups = []
    matched = set()
    for pattern, params_config in layerwise_params.items():
        pattern_params = []
        for name, param in model.named_parameters():
            if pattern.replace('*', '') in name:
                pattern_params.append(param)
                matched.add(name)
        if pattern_params:
            param_groups.append(dict(params=pattern_params, **params_config))
    remaining = [param for name, param in model.named_parameters() if name not in matched]
    if remaining:
        param_groups.append(dict(params=remaining))
    return param_groups

# ===================== COLAB PARAMETRELERI =====================
max_epoch = 30
ignore_index = len(CLASSES)
train_batch_size = 4               # T4 16GB icin guvenli deger
val_batch_size = 4
lr = 4e-4
weight_decay = 0.01
backbone_lr = 4e-4
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

# Agirlik ve log ayarlari
weights_name = "decouplenet-loveda-colab-epoch30"
weights_path = "/content/drive/MyDrive/DecoupleNet/model_weights/loveda"
test_weights_name = "last"
log_name = 'loveda/decouplenet-colab'
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = 1                           # Colab tek GPU

# ===================== TEST AYARLARI =====================
test_weights_name = "decouplenet-loveda-colab-epoch30"  # .ckpt uzantisi olmadan

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


# ===================== VERI SETLERI =====================
train_dataset = LoveDAFlatTrainDataset(
    transform=train_aug,
    data_root='/content/drive/MyDrive/DecoupleNet/data/LoveDA/Train',
    mosaic_ratio=0.25
)

val_dataset = LoveDAFlatTrainDataset(
    transform=val_aug,
    data_root='/content/drive/MyDrive/DecoupleNet/data/LoveDA/Val',
    mosaic_ratio=0.0
)

# Colab icin DataLoader ayarlari
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=2,
    pin_memory=True,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

# ===================== OPTIMIZER =====================
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

# ===================== TEST DATASET =====================
test_dataset = LoveDAFlatTestDataset(
    data_root='/content/drive/MyDrive/DecoupleNet/data/LoveDA/Test'
)
