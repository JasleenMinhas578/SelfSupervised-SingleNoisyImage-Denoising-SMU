# Import necessary libraries
import argparse  # For parsing command-line arguments
import torch  # PyTorch library for tensor operations
from torchvision import transforms  # Provides common image transformations
from PIL import Image  # To handle image input/output
from Denoising import Image_denoise  # Custom image denoising function defined elsewhere

# Function to load and preprocess the input image
def load_image(image_path):
    # Open the image and ensure it's in RGB format
    image = Image.open(image_path).convert("RGB")
    
    # Define transformation: convert the image to a tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to [0,1] float tensor
    ])
    
    # Apply transformation and add batch dimension [1, C, H, W]
    return transform(image).unsqueeze(0)

# Main function to handle argument parsing and invoke denoising
def main():
    # Create argument parser for command-line input
    parser = argparse.ArgumentParser(description="Image Denoising with Self-Supervised Learning")
    
    # Required argument: path to the clean input image
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to clean input image")
    
    # Optional argument: noise level (default is 0.1)
    parser.add_argument("--noise_level", type=float, default=0.1, help="Noise level (standard deviation)")
    
    # Parse the input arguments
    args = parser.parse_args()
    
    # Load the input image as a tensor
    clean_tensor = load_image(args.input)
    
    # Perform denoising using the imported function with specified noise level
    Image_denoise(clean_tensor, noise_level=args.noise_level)

# Entry point of the script
if __name__ == "__main__":
    main()
