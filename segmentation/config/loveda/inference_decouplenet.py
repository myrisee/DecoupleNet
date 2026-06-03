# Inference için minimal config
# LoveDA sınıfları: background(0), building(1), road(2), water(3), barren(4), forest(5), agricultural(6)

num_classes = 7
classes = ('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural')

# Model ayarları
decode_channels = 64
dropout = 0.1
window_size = 8

# Ağırlık yolu (eğitilmiş model varsa)
weights_path = "model_weights/loveda"
test_weights_name = "unetformer-DecoupleNet_D2"  # .ckpt veya .pth
