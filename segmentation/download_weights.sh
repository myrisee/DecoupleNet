#!/bin/bash

# Pre-trained backbone ağırlıkları indirme
mkdir -p segmentation/pretrain_weights
mkdir -p segmentation/model_weights

# DecoupleNet D2 backbone (ImageNet pre-trained)
cd segmentation/pretrain_weights
wget https://github.com/lwCVer/DecoupleNet/releases/download/weights/DecoupleNet_D2.pth

# LoveDA için eğitilmiş model ağırlıkları (varsa)
# Bu linkler README'de verilmemiş, kendi eğitiminiz gerekebilir
