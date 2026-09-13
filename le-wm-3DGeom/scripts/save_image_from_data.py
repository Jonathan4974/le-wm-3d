import os
import numpy as np
from PIL import Image
from utils import load_images_from_episode

def save_images(images_array, output_dir="output_frames"):
    """
    Saves a numpy array of shape (N, H, W, 3) as individual PNG images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Scale float images [0, 1] to uint8 [0, 255] if necessary
    if images_array.max() <= 1.0:
        images_array = (images_array * 255).astype(np.uint8)
    else:
        images_array = images_array.astype(np.uint8)
        
    # Save each frame sequentially
    for idx in range(images_array.shape[0]):
        img = Image.fromarray(images_array[idx])
        # Padded with zeros (e.g., frame_000.png) for proper sorting
        img.save(os.path.join(output_dir, f"frame_{idx:03d}.png"), format="PNG")
        
    print(f"Successfully saved {images_array.shape[0]} images to '{output_dir}'.")

if __name__ == "__main__":
    # Example: Load images from episode 0 and save them as PNGs
    epi_id = 4
    images = load_images_from_episode(epi_id)
    save_images(images, output_dir=f"./data_visualization/images/extracted_frames_episode_{epi_id}")