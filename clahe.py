import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Apply CLAHE to an image.
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Processed image
    """
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE
    return clahe.apply(image)

def process_dataset(input_base_path, output_base_path):
    """
    Process all images in the dataset applying CLAHE augmentation.
    
    Args:
        input_base_path: Path to original images
        output_base_path: Path where augmented images will be saved
    """
    # Create output base directory if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)
    
    # Get all subdirectories (Density1+Benign, Density1+Malignant, etc.)
    subdirs = [d for d in os.listdir(input_base_path) 
              if os.path.isdir(os.path.join(input_base_path, d))]
    
    for subdir in subdirs:
        input_dir = os.path.join(input_base_path, subdir)
        output_dir = os.path.join(output_base_path, subdir)
        
        # Create output subdirectory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process all images in subdirectory
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.dicom'))]
        
        print(f"Processing {subdir}...")
        for image_file in tqdm(image_files):
            input_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, f"clahe_{image_file}")
            
            try:
                # Read image
                img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Error reading image: {input_path}")
                    continue
                
                # Apply CLAHE
                processed_img = apply_clahe(img)
                
                # Save processed image
                cv2.imwrite(output_path, processed_img)
                
            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")

if __name__ == "__main__":
    # Define input and output paths
    input_base_path = "./mass_images"  # Change this to your input path
    output_base_path = "./mass_images_clahe"  # Change this to your desired output path
    
    # Process the dataset
    process_dataset(input_base_path, output_base_path)
    print("Processing complete!")