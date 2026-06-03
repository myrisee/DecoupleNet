"""
DecoupleNet ile Road ve Building Maskeleri Çıkarma Scripti
Kullanım: python extract_road_building.py -i /path/to/image -o /path/to/output
"""

import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
import albumentations as albu
from geoseg.models.UNetFormer_decouplenet import UNetFormer_DecoupleNet_D2
from pathlib import Path


# LoveDA sınıf indeksleri
CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural')
PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
           [159, 129, 183], [0, 255, 0], [255, 195, 128]]

# RGB renkleri (可视化用)
CLASS_COLORS = {
    0: [255, 255, 255],  # background - beyaz
    1: [0, 0, 255],      # building - mavi
    2: [0, 255, 255],    # road - cyan
    3: [255, 0, 0],      # water - kırmızı
    4: [255, 0, 255],    # barren - magenta
    5: [0, 255, 0],      # forest - yeşil
    6: [255, 255, 0]     # agricultural - sarı
}


def mask_to_rgb(mask, palette):
    """Maskeyi RGB formatına çevir"""
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        mask_rgb[mask == class_id] = color
    return mask_rgb


def extract_single_class_mask(mask, class_id):
    """Belirli bir sınıf için binary maske çıkar"""
    binary_mask = (mask == class_id).astype(np.uint8) * 255
    return binary_mask


def extract_road_building_masks(pred_mask):
    """
    Tahmin maskesinden road ve building maskelerini çıkar.

    Args:
        pred_mask: Tahmin maske [H, W] - 0-6 arası sınıf değerleri

    Returns:
        road_mask: Binary road maskesi [H, W] - 0 veya 255
        building_mask: Binary building maskesi [H, W] - 0 veya 255
        combined_mask: Road+Building combined maske [H, W]
    """
    # LoveDA indeksleri: building=1, road=2
    building_mask = extract_single_class_mask(pred_mask, class_id=1)
    road_mask = extract_single_class_mask(pred_mask, class_id=2)

    # Combined maske (hem road hem building)
    combined_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    combined_mask[pred_mask == 1] = 1  # building
    combined_mask[pred_mask == 2] = 2  # road

    return road_mask, building_mask, combined_mask


def load_model(checkpoint_path, device='cuda'):
    """DecoupleNet modelini yükle"""
    num_classes = 7  # LoveDA sınıf sayısı

    # Model oluştur
    model = UNetFormer_DecoupleNet_D2(num_classes=num_classes)

    # Checkpoint yükle
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint yükleniyor: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Lightning checkpoint mi kontrol et
        if 'state_dict' in checkpoint:
            # Lightning checkpoint'ten state_dict çıkar
            state_dict = checkpoint['state_dict']
            # 'model.' önekini kaldır
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Checkpoint yüklendi!")
    else:
        print(f"Uyarı: Checkpoint bulunamadı: {checkpoint_path}")
        print("Rastgele ağırlıklarla başlatılıyor...")

    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, target_size=(1024, 1024)):
    """Görüntüyü ön işle"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]

    # Resize
    transform = albu.Compose([
        albu.Resize(height=target_size[0], width=target_size[1]),
        albu.Normalize()
    ])
    augmented = transform(image=img)
    img_tensor = torch.from_numpy(augmented['image']).permute(2, 0, 1).float()

    return img_tensor, original_size, img


def postprocess_mask(pred, original_size):
    """Tahmin maskesini orijinal boyuta geri dönüştür"""
    pred_np = pred.squeeze().cpu().numpy()
    # Orijinal boyuta resize
    pred_resized = cv2.resize(pred_np, (original_size[1], original_size[0]),
                               interpolation=cv2.INTER_NEAREST)
    return pred_resized.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='DecoupleNet ile Road ve Building Maskeleri Çıkar')
    parser.add_argument('-i', '--image', type=Path, required=True,
                       help='Giriş görüntüsü veya klasör')
    parser.add_argument('-o', '--output', type=Path, required=True,
                       help='Çıktı klasörü')
    parser.add_argument('-c', '--checkpoint', type=str,
                       default='model_weights/loveda/unetformer-DecoupleNet_D2.ckpt',
                       help='Model checkpoint dosyası')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--size', type=int, nargs=2, default=[1024, 1024],
                       help='Model giriş boyutu (H W)')

    args = parser.parse_args()

    # Çıktı klasörünü oluştur
    args.output.mkdir(parents=True, exist_ok=True)

    # Device ayarla
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan device: {device}")

    # Modeli yükle
    model = load_model(args.checkpoint, device)

    # Giriş görüntülerini bul
    image_paths = []
    if args.image.is_file():
        image_paths = [args.image]
    elif args.image.is_dir():
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            image_paths.extend(args.image.glob(ext))

    print(f"Toplam {len(image_paths)} görüntü işlenecek...")

    with torch.no_grad():
        for img_path in image_paths:
            print(f"İşleniyor: {img_path.name}")

            # Ön işleme
            img_tensor, original_size, original_img = preprocess_image(
                img_path, target_size=tuple(args.size))
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Tahmin
            output = model(img_tensor)
            if isinstance(output, tuple):
                output = output[0]  # Aux output varsa ilk öğeyi al

            # Argmax
            pred = torch.argmax(output, dim=1)

            # Post-processing
            pred_mask = postprocess_mask(pred, original_size)

            # Road ve Building maskelerini çıkar
            road_mask, building_mask, combined_mask = extract_road_building_masks(pred_mask)

            # Kaydet
            base_name = img_path.stem

            # Road maskesi (binary)
            cv2.imwrite(str(args.output / f'{base_name}_road.png'),
                       road_mask)

            # Building maskesi (binary)
            cv2.imwrite(str(args.output / f'{base_name}_building.png'),
                       building_mask)

            # Combined mask (RGB - görselleştirme için)
            combined_rgb = mask_to_rgb(combined_mask, [[0,0,0], [0,0,255], [0,255,255]])
            combined_rgb = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(args.output / f'{base_name}_combined.png'),
                       combined_rgb)

            # Tüm sınıflar maskesi
            full_rgb = mask_to_rgb(pred_mask, PALETTE)
            full_rgb = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(args.output / f'{base_name}_full.png'),
                       full_rgb)

            print(f"  ✓ Road maskesi: {base_name}_road.png")
            print(f"  ✓ Building maskesi: {base_name}_building.png")
            print(f"  ✓ Combined mask: {base_name}_combined.png")
            print(f"  ✓ Full mask: {base_name}_full.png")

    print("\nTüm görüntüler işlendi!")


if __name__ == '__main__':
    main()
