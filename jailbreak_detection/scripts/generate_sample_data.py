import numpy as np
from PIL import Image
import os
import random

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def create_clean_image():
    # Create a random clean image
    img = np.random.rand(32, 32) * 255
    img = img.astype(np.uint8)
    return Image.fromarray(img)

def create_adversarial_image():
    # Create an adversarial image with some patterns
    img = np.zeros((32, 32), dtype=np.uint8)
    
    # Add some random patterns
    for _ in range(5):
        x = random.randint(0, 31)
        y = random.randint(0, 31)
        size = random.randint(2, 5)
        img[max(0, x-size):min(32, x+size), max(0, y-size):min(32, y+size)] = 255
    
    return Image.fromarray(img)

def generate_sample_data(num_samples=100):
    clean_dir = os.path.join(PROJECT_ROOT, 'data', 'clean')
    adv_dir = os.path.join(PROJECT_ROOT, 'data', 'adversarial')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)
    
    # Generate clean images
    for i in range(num_samples):
        img = create_clean_image()
        img.save(os.path.join(clean_dir, f'clean_{i}.png'))
    
    # Generate adversarial images
    for i in range(num_samples):
        img = create_adversarial_image()
        img.save(os.path.join(adv_dir, f'adv_{i}.png'))
    
    print(f"Generated {num_samples} clean and {num_samples} adversarial images")

if __name__ == "__main__":
    generate_sample_data() 