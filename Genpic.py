import torch
from torchvision.utils import save_image
from vae import IMAGE_SIZE, celeb_transform, VAE
from PIL import Image

MODEL_FILE = './checkpoints_best/vae_model_20.pth'

# Load model
model = torch.load(MODEL_FILE, map_location='cpu',weights_only=False)
model.eval()  # Set model to evaluation mode

# Load and preprocess image
image = Image.open('./tested_data/000007.jpg')
image = celeb_transform(image).unsqueeze(0)

# Encode the image to get mu and log_var
mu, log_var = model.encode(image)

# Generate 10 different variations
generated_images = []
for i in range(10):
    w = torch.rand(1) * 0.01  # Introduce a larger variance scaling factor
    std = torch.exp(w * log_var)
    eps = torch.randn_like(std)
    z = eps * std + mu
    # z = model.reparameterize(mu, log_var)  # Sample different latent representations
    recon = model.decode(z)  # Decode to generate an image
    pic = recon.view(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    generated_images.append(pic)

# Optionally, save all images in a grid
save_image(torch.cat(generated_images, dim=0), 'generated_images_grid_000007.jpg', nrow=7)