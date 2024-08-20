#%%
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
# %%
DATA_DIR = './Faces'

dosyalar = os.listdir(DATA_DIR)
toplam_gorsel_sayisi = {}
jpg_formatindaki_gorsel_sayisi = {}

for d in dosyalar:
  dosya_yolu = os.path.join(DATA_DIR, d)
  if os.path.isdir(dosya_yolu):
    tum_gorseller = glob.glob(os.path.join(dosya_yolu, '*'))
    jpg_gorseller = glob.glob(os.path.join(dosya_yolu, '*.jpg'))

    toplam_gorsel_sayisi[d] = len(tum_gorseller)
    jpg_formatindaki_gorsel_sayisi[d] = len(jpg_gorseller)

print('Toplam Görsel Sayısı: ', toplam_gorsel_sayisi)
print('JPG Formatındaki Görsel Sayısı: ', jpg_formatindaki_gorsel_sayisi)
# %%
gorseller = [Image.open(gorsel_yolu) for gorsel_yolu in glob.glob(os.path.join(DATA_DIR, '*/*.jpg'))]

stats = [np.array(img) for img in gorseller]
# %%
fig, axes = plt.subplots(7, 7, figsize=(6, 6))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(gorseller[i])
    ax.axis('off')
# %%
f = list(toplam_gorsel_sayisi.keys())
c = list(toplam_gorsel_sayisi.values())

plt.figure(figsize = (8, 6))
plt.bar(f, c, color = 'skyblue')
plt.xlabel('Dosya İsimleri')
plt.ylabel('Görsel Sayısı')
plt.title('Görsellerin Veri Setine Dağılımı')
plt.xticks(rotation = 45, ha = 'right')
plt.legend()
# %%
f = list(jpg_formatindaki_gorsel_sayisi.keys())
c = list(jpg_formatindaki_gorsel_sayisi.values())

plt.figure(figsize = (8, 6))
plt.bar(f, c, color = 'red')
plt.xlabel('Dosya İsimleri')
plt.ylabel('Görsel Sayısı')
plt.title('Görsellerin Veri Setine Dağılımı')
plt.xticks(rotation = 45, ha = 'right')
plt.legend()
# %%
def ilk_12_gorseli_goster(ilk_12_gorsel, cols = 4):
  n_ = len(ilk_12_gorsel)
  rows = n_ // cols + int(n_ % cols > 0)

  plt.figure(figsize = (6, 6))

  for i, image in enumerate(ilk_12_gorsel):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(image)
    plt.axis('off')
  plt.legend()
# %%
kate_winslet_ = "./Faces/Angelina Jolie"
kate_winslet_jpg = glob.glob(os.path.join(kate_winslet_, '*.jpg'))

ilk_12_gorsel = [Image.open(img_path) for img_path in kate_winslet_jpg[:12]]
ilk_12_gorseli_goster(ilk_12_gorsel)
# %%
jognny_depp_ = "./Faces/Johnny Depp"
jognny_depp_jpg = glob.glob(os.path.join(jognny_depp_, '*.jpg'))

ilk_12_gorsel = [Image.open(img_path) for img_path in jognny_depp_jpg[:12]]
ilk_12_gorseli_goster(ilk_12_gorsel)
# %%
jr = "./Faces/Robert Downey Jr"
jr_jpg = glob.glob(os.path.join(jr, '*.jpg'))

ilk_12_gorsel = [Image.open(img_path) for img_path in jr_jpg[:12]]
ilk_12_gorseli_goster(ilk_12_gorsel)
# %%
scarlett_ = "./Faces/Scarlett Johansson"
scarlett_jpg = glob.glob(os.path.join(scarlett_, '*.jpg'))

ilk_12_gorsel = [Image.open(img_path) for img_path in scarlett_jpg[:12]]
ilk_12_gorseli_goster(ilk_12_gorsel)
# %%
def gorsel_bilgileri_(f, num_s = 5):
  i_p = glob.glob(os.path.join(f, '*.jpg'))
  random_sample = random.sample(i_p, num_s)
  gorsel_b = []

  for img_path in random_sample:
    with Image.open(img_path) as img:
      genislik, yukseklik = img.size
      kanal_sayisi = img.mode
      gorsel_b.append({
          'dosya_yolu' : img_path,
          'genislik' : genislik,
          'yukseklik' : yukseklik,
          'kanal' : kanal_sayisi,
          'kanal_sayisi' : len(img.getbands())
      })
  return gorsel_b
# %%
scarlett_ = "./Faces/Scarlett Johansson"
gorseller_hakkinda = gorsel_bilgileri_(scarlett_, num_s = 5)

for info in gorseller_hakkinda:
    print(f"Dosya Yolu: {info['dosya_yolu']}")
    print(f"Genişlik: {info['genislik']}")
    print(f"Yükseklik: {info['yukseklik']}")
    print(f"Kanal: {info['kanal']} (Kanal Sayısı: {info['kanal_sayisi']})")
    print("-" * 30)
# %%
jr = "./Faces/Robert Downey Jr"
gorseller_hakkinda = gorsel_bilgileri_(jr, num_s = 5)

for info in gorseller_hakkinda:
    print(f"Dosya Yolu: {info['dosya_yolu']}")
    print(f"Genişlik: {info['genislik']}")
    print(f"Yükseklik: {info['yukseklik']}")
    print(f"Kanal: {info['kanal']} (Kanal Sayısı: {info['kanal_sayisi']})")
    print("-" * 30)