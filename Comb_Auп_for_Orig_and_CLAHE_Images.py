import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from tqdm import tqdm

def get_mammography_augmentation(angle):
    """
    Creates an augmentation pipeline for specific rotation angle
    """
    return A.Compose([
        A.Rotate(limit=[angle-1, angle+1], p=1.0),  # Конкретный угол поворота
        A.PadIfNeeded(
            min_height=None,
            min_width=None,
            pad_height_divisor=32,
            pad_width_divisor=32,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=1.0
        )
    ], p=1.0)

def process_and_augment_images(orig_base_path, clahe_base_path, output_base_path):
    """
    Process both original and CLAHE images with augmentations
    """
    # Определяем все углы поворота
    rotation_angles = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(orig_base_path) 
              if os.path.isdir(os.path.join(orig_base_path, d))]
    
    for subdir in subdirs:
        print(f"\nProcessing {subdir}")
        
        output_dir = os.path.join(output_base_path, subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        orig_dir = os.path.join(orig_base_path, subdir)
        clahe_dir = os.path.join(clahe_base_path, subdir)
        
        # Get all image files
        image_files = [f for f in os.listdir(orig_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        
        for img_file in tqdm(image_files, desc="Augmenting images"):
            # Read images
            orig_path = os.path.join(orig_dir, img_file)
            clahe_path = os.path.join(clahe_dir, f"clahe_{img_file}")
            
            orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
            clahe_img = cv2.imread(clahe_path, cv2.IMREAD_GRAYSCALE)
            
            if orig_img is None or clahe_img is None:
                print(f"Error reading images for {img_file}")
                continue
            
            # Save original versions
            cv2.imwrite(os.path.join(output_dir, f"orig_{img_file}"), orig_img)
            cv2.imwrite(os.path.join(output_dir, f"clahe_{img_file}"), clahe_img)
            
            # Apply rotations
            for angle in rotation_angles:
                transform = get_mammography_augmentation(angle)
                
                # Augment original image
                aug_orig = transform(image=orig_img)['image']
                cv2.imwrite(os.path.join(output_dir, f"rotation_{angle}_orig_{img_file}"), aug_orig)
                
                # Augment CLAHE image
                aug_clahe = transform(image=clahe_img)['image']
                cv2.imwrite(os.path.join(output_dir, f"rotation_{angle}_clahe_{img_file}"), aug_clahe)
                
                # Apply flips
                # Horizontal flip
                h_flip_orig = cv2.flip(aug_orig, 1)
                h_flip_clahe = cv2.flip(aug_clahe, 1)
                cv2.imwrite(os.path.join(output_dir, f"rotation_{angle}_orig_hflip_{img_file}"), h_flip_orig)
                cv2.imwrite(os.path.join(output_dir, f"rotation_{angle}_clahe_hflip_{img_file}"), h_flip_clahe)
                
                # Vertical flip
                v_flip_orig = cv2.flip(aug_orig, 0)
                v_flip_clahe = cv2.flip(aug_clahe, 0)
                cv2.imwrite(os.path.join(output_dir, f"rotation_{angle}_orig_vflip_{img_file}"), v_flip_orig)
                cv2.imwrite(os.path.join(output_dir, f"rotation_{angle}_clahe_vflip_{img_file}"), v_flip_clahe)

if __name__ == "__main__":
    # Define paths
    orig_base_path = "./mass_images"
    clahe_base_path = "./mass_images_clahe"
    output_base_path = "./augmented_dataset"
   
    process_and_augment_images(orig_base_path, clahe_base_path, output_base_path)
    print("\nAugmentation completed!")  