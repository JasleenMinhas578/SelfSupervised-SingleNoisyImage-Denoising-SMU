# Import required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For displaying progress bar during training
from torchvision import transforms  # Common image transformations
from PIL import Image  # For image handling
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Functional API for neural nets

# Define image transformation: converts PIL image to PyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Define a custom activation function: Smooth MU (SMU)
class SMU(nn.Module):
    def __init__(self, alpha=0.25):
        super(SMU, self).__init__()
        self.alpha = alpha
        # Learnable parameter controlling smoothness
        self.mu = torch.nn.Parameter(torch.tensor(1000000.0))  

    def forward(self, x):
        # Smooth nonlinear transformation applied to x
        return ((1 + self.alpha) * x + (1 - self.alpha) * x * torch.erf(self.mu * (1 - self.alpha) * x)) / 2

# Define the neural network architecture
class network(nn.Module):
    def __init__(self, n_chan, chan_embed=48):
        super(network, self).__init__()
        self.act = SMU()  # Using SMU activation function
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)  # First convolution layer
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)  # Second convolution layer
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)  # Output layer (1x1 convolution)

    def forward(self, x):
        # Apply convolutions with activation
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x

# Downsamples the input image in two patterns using learnable filters
def pair_downsampler(img):
    c = img.shape[1]  # Get number of channels

    # Create two custom 2x2 downsampling filters
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    # Apply filters with stride 2 to perform downsampling
    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2

# Mean Squared Error loss
def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.mean((gt - pred) ** 2)

# Self-supervised loss function
def loss_func(noisy_img, model):
    # Downsample noisy image into two views
    noisy1, noisy2 = pair_downsampler(noisy_img)

    # Predict noise and subtract it from the noisy image
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)

    # Residual consistency loss (between cross predictions)
    loss_res = 0.5 * (mse(noisy1, pred2) + mse(noisy2, pred1))

    # Denoise the full image
    noisy_denoised = noisy_img - model(noisy_img)

    # Downsample the denoised image
    denoised1, denoised2 = pair_downsampler(noisy_denoised)

    # Consistency loss between residuals and denoised views
    loss_cons = 0.5 * (mse(pred1, denoised1) + mse(pred2, denoised2))

    return loss_res + loss_cons  # Total loss

# Training step for a single epoch
def train(model, optimizer, noisy_img):
    loss = loss_func(noisy_img, model)  # Compute loss
    optimizer.zero_grad()  # Clear gradients
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model parameters
    return loss.item()  # Return loss value

# Inference: denoise the image using the trained model
def denoise(model, noisy_img):
    with torch.no_grad():  # Disable gradient computation
        # Predict noise and subtract it, clamp values between 0 and 1
        return torch.clamp(noisy_img - model(noisy_img), 0, 1)

# Compute PSNR between original and denoised image
def calculate_psnr(clean, denoised):
    mse = torch.mean((clean - denoised) ** 2)
    return 10 * torch.log10(1 / mse).item()  # Convert to dB

# Complete denoising pipeline
def Image_denoise(clean_tensor, noise_level=0.1, show_results=True):
    # Add synthetic Gaussian noise
    noisy_tensor = clean_tensor + torch.randn_like(clean_tensor) * noise_level
    noisy_tensor = torch.clamp(noisy_tensor, 0, 1)  # Keep values in [0,1]
    
    # Move tensors to appropriate device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noisy_img = noisy_tensor.to(device)
    
    # Initialize the denoising model and optimizer
    n_chan = noisy_img.shape[1]
    model = network(n_chan).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)  # Learning rate scheduler
    
    # Training loop
    for epoch in tqdm(range(2000)):
        train(model, optimizer, noisy_img)
        scheduler.step()  # Update learning rate
    
    # Generate denoised output
    denoised_tensor = denoise(model, noisy_img)
    
    # Convert tensors to NumPy arrays for visualization
    clean_np = clean_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    noisy_np = noisy_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
    denoised_np = denoised_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    
    # Compute PSNR values for comparison
    psnr_noisy = calculate_psnr(clean_tensor.to(device), noisy_img)
    psnr_denoised = calculate_psnr(clean_tensor.to(device), denoised_tensor)
    
    # Plot the clean, noisy, and denoised images side by side
    if show_results:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(clean_np)
        plt.title(f'Clean Image\nPSNR: ∞ dB')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(noisy_np)
        plt.title(f'Noisy Image\nPSNR: {psnr_noisy:.2f} dB')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(denoised_np)
        plt.title(f'Denoised Image\nPSNR: {psnr_denoised:.2f} dB')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return denoised_np  # Return denoised image as NumPy array
