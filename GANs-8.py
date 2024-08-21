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
#%%
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#%%
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transformations = transforms.Compose([transforms.Resize((64)),transforms.CenterCrop(64), transforms.ToTensor(), transforms.Normalize(*stats),])
dataset = ImageFolder(DATA_DIR, transform = transformations)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)

#%%
def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    tensor = tensor * std + mean
    return tensor

def show_images(dataloader, nmax=64):
    images, _ = next(iter(dataloader))
    images = images.detach().cpu()  # CPU'ya taşı ve detach et
    # Unnormalize işlemi
    images = unnormalize(images, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    images = torch.clamp(images, 0, 1)  # [0, 1] aralığına getir
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(images[:nmax], nrow=8).permute(1, 2, 0))
    plt.show()

show_images(dataloader)

# %%
def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]
# %%
def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break
# %%
show_batch(dataloader)

#%%
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
#%%
device = get_default_device()
device

#%%
dataloader= DeviceDataLoader(dataloader, device)

# %%
discriminator = nn.Sequential(
  nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
  nn.BatchNorm2d(64),
  nn.LeakyReLU(0.2, inplace=True),
  # out: 64 x 32 x 32

  nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
  nn.BatchNorm2d(128),
  nn.LeakyReLU(0.2, inplace=True),
  # out: 128 x 16 x 16

  nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
  nn.BatchNorm2d(256),
  nn.LeakyReLU(0.2, inplace=True),
  # out: 256 x 8 x 8

  nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
  nn.BatchNorm2d(512),
  nn.LeakyReLU(0.2, inplace=True),
  # out: 512 x 4 x 4

  nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
    # out: 1 x 1 x 1

    nn.Flatten(),
    nn.Sigmoid()
)

discriminator = to_device(discriminator, device)


#%%
latent_size = 128
generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    # out: 64 x 32 x 32
     nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x 64 x 64
)
generator = to_device(generator, device)


#%%
# Discriminator Training
def train_discriminator(real_images, opt_d):
  # Clear discriminator gradients
  opt_d.zero_grad()

  # Pass real images through discriminator
  real_preds = discriminator(real_images)
  real_targets = torch.ones(real_images.size(0), 1, device=device)
  real_loss = nn.functional.binary_cross_entropy(real_preds, real_targets)
  real_score = torch.mean(real_preds).item()
  
  # Generate fake images
  latent = torch.randn(256, latent_size, 1, 1, device=device)
  fake_images = generator(latent)

  # Pass fake images through discriminator
  fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
  fake_preds = discriminator(fake_images)
  fake_loss = nn.functional.binary_cross_entropy(fake_preds, fake_targets)
  fake_score = torch.mean(fake_preds).item()

  # Update discriminator weights
  loss = real_loss + fake_loss
  loss.backward()
  opt_d.step()
  return loss.item(), real_score, fake_score

#%% Generator Training
def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate fake images
    latent = torch.randn(256, latent_size, 1, 1, device=device)
    fake_images = generator(latent)
    
    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(256, 1, device=device)
    loss = nn.functional.binary_cross_entropy(preds, targets)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()

#%%
from torchvision.utils import save_image
sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)


def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

#%%
fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
save_samples(0, fixed_latent)

#%%
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for real_images, _ in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            loss_g = train_generator(opt_g)
    # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
    
        # Save generated images
        save_samples(epoch+start_idx, fixed_latent, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores

#%%
lr = 0.005
epochs = 400
history = fit(epochs, lr)
#%%
losses_g, losses_d, real_scores, fake_scores = history
torch.save(generator.state_dict(), 'G.pth')
torch.save(discriminator.state_dict(), 'D.pth')

#%%
from IPython.display import Image
Image('./generated/generated-images-0001.png')

#%%
Image('./generated/generated-images-0010.png')

#%%
Image('./generated/generated-images-0050.png')

#%%
Image('./generated/generated-images-0100.png')

#%%
Image('./generated/generated-images-0200.png')

#%%
Image('./generated/generated-images-0300.png')

#%%
Image('./generated/generated-images-0350.png')

#%%
Image('./generated/generated-images-0400.png')

#%%
import cv2
import os

vid_fname = 'gans_training.avi'

files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'generated' in f]
files.sort()

out = cv2.VideoWriter(vid_fname,cv2.VideoWriter_fourcc(*'MP4V'), 1, (530,530))
[out.write(cv2.imread(fname)) for fname in files]
out.release()


#%%
plt.plot(losses_d, '-')
plt.plot(losses_g, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses');

#%%
plt.plot(real_scores, '-')
plt.plot(fake_scores, '-')
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend(['Real', 'Fake'])
plt.title('Scores');

# %%
