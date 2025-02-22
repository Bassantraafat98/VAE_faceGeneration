import torch
from torch import nn
from torchvision import transforms

# Define constants for image size and latent dimension
IMAGE_SIZE = 150
LATENT_DIM = 128
image_dim = 3 * IMAGE_SIZE * IMAGE_SIZE  # Total number of pixels for a color image

# Transformation for images before feeding into the VAE
celeb_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, antialias=True),  # Resize image to IMAGE_SIZE
    transforms.CenterCrop(IMAGE_SIZE),  # Crop the center of the image
    transforms.ToTensor()  # Convert image to tensor format
])
# Transformation for images after decoding
celeb_transform1 = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, antialias=True),  # Resize image to IMAGE_SIZE
    transforms.CenterCrop(IMAGE_SIZE)  # Crop the center of the image
])

# Define the Variational Autoencoder (VAE) class
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Hidden dimensions for the encoder and decoder
        hidden_dims = [32, 64, 128, 256, 512]
        self.final_dim = hidden_dims[-1]  # Final dimension for the encoder output
        in_channels = 3  # Number of input channels (RGB)
        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),  # Convolutional layer
                    nn.BatchNorm2d(h_dim),  # Batch normalization
                    nn.LeakyReLU()  # Activation function
                )
            )
            in_channels = h_dim  # Update input channels for the next layer
        self.encoder = nn.Sequential(*modules)  # Combine all encoder layers

        # Determine the output size after the encoder
        out = self.encoder(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.size = out.shape[2]  # Output spatial size after encoder
        # Define linear layers for mean and variance of the latent space
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.size * self.size, LATENT_DIM)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.size * self.size, LATENT_DIM)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(LATENT_DIM, hidden_dims[-1] * self.size * self.size)  # Input layer for decoder
        hidden_dims.reverse()  # Reverse hidden dimensions for decoding
        # Create decoder layers
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),  # Transposed convolutional layer
                    nn.BatchNorm2d(hidden_dims[i + 1]),  # Batch normalization
                    nn.LeakyReLU()  # Activation function
                )
            )
        self.decoder = nn.Sequential(*modules)  # Combine all decoder layers
        # Final layer to produce output image
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),  # Transposed convolutional layer
            nn.BatchNorm2d(hidden_dims[-1]),  # Batch normalization
            nn.LeakyReLU(),  # Activation function
            nn.Conv2d(hidden_dims[-1], out_channels=3,  # Final convolution to output 3 channels (RGB)
                      kernel_size=3, padding=1),
            nn.Sigmoid()  # Sigmoid activation to ensure output is between 0 and 1
        )

    # Encode input images to latent space
    def encode(self, x):
        result = self.encoder(x)  # Pass through encoder
        result = torch.flatten(result, start_dim=1)  # Flatten the output
        mu = self.fc_mu(result)  # Mean of latent space
        log_var = self.fc_var(result)  # Log variance of latent space
        return mu, log_var
    
    # Reparameterization trick to sample from the latent space
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)  # Random noise
        return eps * std + mu  # Sample from the distribution
    
    # Decode latent variable back to image
    def decode(self, z):
        result = self.decoder_input(z)  # Pass through the input layer of the decoder
        result = result.view(-1, self.final_dim, self.size, self.size)  # Reshape for decoder
        result = self.decoder(result)  # Pass through decoder
        result = self.final_layer(result)  # Final layer to produce output
        result = celeb_transform1(result)  # Apply transformation to final output
        result = torch.flatten(result, start_dim=1)  # Flatten the output
        result = torch.nan_to_num(result)  # Replace NaNs with zero
        return result
    
    # Forward pass through the VAE
    def forward(self, x):
        mu, log_var = self.encode(x)  # Encode input
        z = self.reparameterize(mu, log_var)  # Sample latent variable
        return self.decode(z), mu, log_var  # Decode and return output, mean, and log variance