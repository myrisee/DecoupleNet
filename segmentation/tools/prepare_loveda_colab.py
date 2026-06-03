"""
LoveDA Veri Seti Hazirlama Scripti — HuggingFace -> Egitim Formatina Donusturme

Bu script su islemleri yapar:
1. HuggingFace'den (chloechia/loveda) veri setini indirir
2. Dosyalari egitim pipeline'inin bekledigi flat dizin yapisina tasir
3. Maske donusumunu uygular (orijinal degerler -> egitim degerleri)

Kullanim (Colab hucresinde):
    %cd /content/DecoupleNet/segmentation
    %run tools/prepare_loveda_colab.py

Veya:
    python tools/prepare_loveda_colab.py --data-root /content/data/LoveDA
"""

import os
import sys
import glob
import shutil
import argparse
import time
import numpy as np
import cv2
from pathlib import Path

# HuggingFace indirme icin
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub kuruluyor...")
    os.system("pip install -q huggingface_hub")
    from huggingface_hub import snapshot_download


# ===================== AYARLAR =====================
HF_REPO_ID = "chloechia/loveda"
HF_REPO_TYPE = "dataset"

# HuggingFace'deki dizin isimleri (orijinal)
HF_TRAIN_IMAGES_DIR = "urban:rural train images"
HF_TRAIN_MASKS_DIR = "urban:rural train masks"
HF_VAL_IMAGES_DIR = "urban:rural val images"
HF_VAL_MASKS_DIR = "urban:rural val masks"


def convert_label(mask):
    """Orijinal LoveDA maskelerini egitim formatina donustur.

    Orijinal: 0=no-data/ignore, 1=background, 2=building, ..., 7=agriculture
    Hedef:    0=background, 1=building, 2=road, ..., 6=agriculture, 7=ignore
    """
    mask[mask == 0] = 8
    mask -= 1
    return mask


def convert_masks(masks_dir, output_dir):
    """Tum maskeleri donustur ve kaydet."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '_rgb', exist_ok=True)

    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    total = len(mask_paths)
    print(f"  Maske donusumu: {total} dosya isleniyor...")

    for i, mask_path in enumerate(mask_paths):
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        label = convert_label(mask)

        # Gri tonlamali maske kaydet
        out_mask_path = os.path.join(output_dir, f"{mask_filename}.png")
        cv2.imwrite(out_mask_path, label)

        # RGB gorsellesrtirme kaydet (opsiyonel)
        PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
                   [159, 129, 183], [0, 255, 0], [255, 195, 128]]
        h, w = label.shape[0], label.shape[1]
        mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        mask_convert = label[np.newaxis, :, :]
        for cls_idx, color in enumerate(PALETTE):
            mask_rgb[np.all(mask_convert == cls_idx, axis=0)] = color
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        out_rgb_path = os.path.join(output_dir + '_rgb', f"{mask_filename}.png")
        cv2.imwrite(out_rgb_path, mask_rgb)

        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"    [{i+1}/{total}] islendi")

    print(f"  Tamamlandi: {total} maske donusturuldu")
    return total


def download_and_organize(data_root):
    """HuggingFace'den indir ve hedef dizin yapisina tasir."""

    print("=" * 60)
    print("LoveDA Veri Seti Hazirlama — HuggingFace Kaynagi")
    print("=" * 60)

    # Hedef dizinleri olustur
    train_img_dir = os.path.join(data_root, "Train", "images_png")
    train_mask_dir = os.path.join(data_root, "Train", "masks_png")
    val_img_dir = os.path.join(data_root, "Val", "images_png")
    val_mask_dir = os.path.join(data_root, "Val", "masks_png")

    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        os.makedirs(d, exist_ok=True)

    # ===================== INDIRME =====================
    download_configs = [
        (HF_TRAIN_IMAGES_DIR, train_img_dir, "Train Goruntuleri"),
        (HF_TRAIN_MASKS_DIR, train_mask_dir, "Train Maskeleri"),
        (HF_VAL_IMAGES_DIR, val_img_dir, "Val Goruntuleri"),
        (HF_VAL_MASKS_DIR, val_mask_dir, "Val Maskeleri"),
    ]

    total_downloaded = 0

    for hf_dir, target_dir, description in download_configs:
        print(f"\n>>> {description} indiriliyor...")
        t0 = time.time()

        # snapshot_download ile sadece ilgili dizini indir
        allow_pattern = [f"{hf_dir}/*.png"]
        cache_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            allow_patterns=allow_pattern,
        )

        # Indirilen dosyalari hedef dizine tasi
        src_dir = os.path.join(cache_path, hf_dir)
        if not os.path.isdir(src_dir):
            print(f"  HATA: {src_dir} bulunamadi!")
            print(f"  Mevcut dizinler: {os.listdir(cache_path)}")
            continue

        files = [f for f in os.listdir(src_dir) if f.endswith('.png')]
        for f in files:
            src = os.path.join(src_dir, f)
            dst = os.path.join(target_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        count = len(files)
        total_downloaded += count
        elapsed = time.time() - t0
        print(f"  {count} dosya indirildi ({elapsed:.1f}s)")

    print(f"\n>>> Toplam indirilen: {total_downloaded} dosya")

    # ===================== MASKE DONUSUMU =====================
    print("\n>>> Maske donusumu baslatiliyor...")

    print("\n[1/2] Train maskeleri donusturuluyor:")
    train_count = convert_masks(
        train_mask_dir,
        os.path.join(data_root, "Train", "masks_png_convert")
    )

    print("\n[2/2] Val maskeleri donusturuluyor:")
    val_count = convert_masks(
        val_mask_dir,
        os.path.join(data_root, "Val", "masks_png_convert")
    )

    # ===================== DOGRULAMA =====================
    print("\n" + "=" * 60)
    print("DOGRULAMA SONUCLARI")
    print("=" * 60)

    train_img_count = len(glob.glob(os.path.join(train_img_dir, "*.png")))
    train_mask_count = len(glob.glob(os.path.join(data_root, "Train", "masks_png_convert", "*.png")))
    val_img_count = len(glob.glob(os.path.join(val_img_dir, "*.png")))
    val_mask_count = len(glob.glob(os.path.join(data_root, "Val", "masks_png_convert", "*.png")))

    print(f"  Train goruntuleri:     {train_img_count}")
    print(f"  Train maskeleri:      {train_mask_count}")
    print(f"  Val goruntuleri:       {val_img_count}")
    print(f"  Val maskeleri:        {val_mask_count}")

    # Eslesme kontrolu
    checks = [
        ("Train", train_img_count, train_mask_count),
        ("Val", val_img_count, val_mask_count),
    ]

    all_ok = True
    for name, img_c, mask_c in checks:
        if img_c == mask_c and img_c > 0:
            print(f"  {name}: OK ({img_c} eslesme)")
        else:
            print(f"  {name}: HATA! Goruntu={img_c}, Maske={mask_c}")
            all_ok = False

    if all_ok:
        print("\nTum kontroller basarili! Egitime hazirsiniz.")
    else:
        print("\nUYARI: Bazi kontroller basarisiz oldu. Lutfen kontrol edin.")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="LoveDA veri setini HuggingFace'den indir ve hazirla")
    parser.add_argument("--data-root", default="/content/drive/MyDrive/DecoupleNet/data/LoveDA",
                        help="Veri setinin kaydedilecegi dizin (varsayilan: /content/data/LoveDA)")
    args = parser.parse_args()

    success = download_and_organize(args.data_root)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
