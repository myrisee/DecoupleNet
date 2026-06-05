"""
Local inference script — CPU/AMD GPU icin.
Supervision_Train veya tam config gerektirmez, direkt model agirliklarini yukler.

Kullanim:
  python local_inference.py --ckpt model_weights/loveda/decouplenet-loveda-colab-epoch30.ckpt \
                            --data-root data/LoveDA/Test \
                            --output fig_results/loveda/local_test
"""

import argparse
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as albu
from PIL import Image
from tqdm import tqdm
from pathlib import Path


PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
           [159, 129, 183], [0, 255, 0], [255, 195, 128]]


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]
    return mask_rgb


def get_test_images(data_root):
    """LoveDA Test klasor yapisi: Test/Urban/images_png/ ve Test/Rural/images_png/"""
    img_ids = []
    for scene in ['Urban', 'Rural']:
        img_dir = os.path.join(data_root, scene, 'images_png')
        if os.path.isdir(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith('.png'):
                    img_ids.append((os.path.join(img_dir, fname), scene, fname.split('.')[0]))
    return img_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint yolu (.ckpt)")
    parser.add_argument("--data-root", type=str, default="data/LoveDA/Test", help="Test veri kok dizini")
    parser.add_argument("--output", type=str, default="fig_results/loveda/local_test", help="Cikti klasoru")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--rgb", action="store_true", help="Renkli maske ciktisi")
    args = parser.parse_args()

    # Cihaz secimi (AMD GPU -> CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    # Modeli olustur
    from geoseg.models.UNetFormer_decouplenet import UNetFormer_DecoupleNet_D2
    model = UNetFormer_DecoupleNet_D2(num_classes=7)

    # Lightning checkpoint'ten agirliklari yukle
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model yuklendi: {args.ckpt}")

    # Test goruntulerini listele
    img_list = get_test_images(args.data_root)
    print(f"Toplam test goruntusu: {len(img_list)}")

    if len(img_list) == 0:
        print(f"HATA: {args.data_root} altinda goruntu bulunamadi!")
        print("Beklenen yapi: data_root/Urban/images_png/ ve data_root/Rural/images_png/")
        return

    # Cikti klasorleri
    for scene in ['Urban', 'Rural']:
        os.makedirs(os.path.join(args.output, scene), exist_ok=True)

    # Normalization transform
    transform = albu.Normalize()

    # Inference
    total_time = 0
    with torch.no_grad():
        for img_path, scene, img_id in tqdm(img_list, desc="Inference"):
            img = np.array(Image.open(img_path).convert('RGB'))
            augmented = transform(image=img)
            img_tensor = torch.from_numpy(augmented['image']).permute(2, 0, 1).unsqueeze(0).float().to(device)

            t0 = time.time()
            raw_pred = model(img_tensor)
            if isinstance(raw_pred, (list, tuple)):
                raw_pred = raw_pred[0]
            pred = nn.Softmax(dim=1)(raw_pred).argmax(dim=1).squeeze(0).cpu().numpy()
            total_time += time.time() - t0

            # Kaydet
            if args.rgb:
                mask_rgb = label2rgb(pred.astype(np.uint8))
                mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(args.output, scene, img_id + '.png'), mask_rgb)
            else:
                cv2.imwrite(os.path.join(args.output, scene, img_id + '.png'), pred.astype(np.uint8))

    print(f"\nTamamlandi! {len(img_list)} goruntu islenmedi.")
    print(f"Toplam sure: {total_time:.1f}s ({total_time/len(img_list):.2f}s/goruntu)")
    print(f"Cikti: {args.output}/")


if __name__ == "__main__":
    main()
