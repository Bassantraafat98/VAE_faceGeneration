# Variational Autoencoder (VAE) Image Generator

This project implements a Variational Autoencoder (VAE) for image generation and manipulation using PyTorch. The implementation consists of two main files: `vae.py` containing the VAE model architecture, and `genpic.py` for generating image variations.

## Architecture Overview

### VAE Model (`vae.py`)

The VAE implementation features:
- Input image size: 150x150 pixels
- Latent dimension: 128
- Encoder: Progressive convolutional layers (32 → 64 → 128 → 256 → 512 channels)
- Decoder: Reverse architecture with transposed convolutions
- Activation: LeakyReLU
- Final activation: Sigmoid
- Normalization: BatchNorm2d

Key Components:
1. **Encoder**:
   - Sequential convolutional layers with increasing channel dimensions
   - BatchNormalization and LeakyReLU activation
   - Linear layers for generating mean (μ) and log variance

2. **Decoder**:
   - Linear layer to project from latent space
   - Transposed convolutions for upsampling
   - Final sigmoid activation for image reconstruction

3. **Image Transforms**:
   - `celeb_transform`: Preprocessing for input images (resize, center crop, tensor conversion)
   - `celeb_transform1`: Post-processing for decoded images

### Image Generator (`genpic.py`)

The generator script provides functionality to:
- Load a trained VAE model
- Process input images
- Generate multiple variations using the latent space
- Save the generated images in a grid format

## Usage

### Model Training

The model should be trained and saved to `./checkpoints_best/vae_model_20.pth`

### Generating Images

1. Place your input image in `./data/tested_data/`
2. Run the generator:
```bash
python genpic.py
```

The script will:
- Load the model from the specified checkpoint
- Process the input image
- Generate 10 variations
- Save the results as 'generated_images_grid2.jpg'

## Parameters

### VAE Parameters:
- `IMAGE_SIZE`: 150 (pixels)
- `LATENT_DIM`: 128
- Hidden dimensions: [32, 64, 128, 256, 512]

### Generation Parameters:
- Number of variations: 10
- Variance scaling factor: 0.01

## Dependencies

- PyTorch
- torchvision
- PIL (Python Imaging Library)

## File Structure

```
.
├── vae.py              # VAE model implementation
├── genpic.py           # Image generation script
├── checkpoints_best/   # Model checkpoints directory
└── data/
    └── tested_data/   # Input images directory
```

## Implementation Details

### Reparameterization

The VAE uses the reparameterization trick to enable backpropagation through the sampling process:
```python
z = eps * std + mu
```
where:
- eps is sampled from N(0,1)
- std is computed as exp(0.5 * log_var)
- mu is the encoded mean

### Image Generation Process

1. Encode input image to get μ and log_var
2. Sample points in latent space using controlled variance
3. Decode samples to generate variations
4. Combine generated images into a grid

## Notes

- The model expects RGB images (3 channels)
- Generated images are saved in the project root directory
- Adjust the variance scaling factor (w) in genpic.py to control variation intensity
