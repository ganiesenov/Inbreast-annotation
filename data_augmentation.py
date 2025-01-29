import albumentations as A
import cv2

def get_mammography_augmentation():
    """
    Creates an augmentation pipeline specifically designed for mammography images,
    following the paper's methodology with some additional careful augmentations.
    """
    return A.Compose([
        # 1. Multi-angle rotation as specified in the paper
        A.OneOf([
            A.Rotate(limit=[29, 31], p=0.1),      # ~30°
            A.Rotate(limit=[59, 61], p=0.1),      # ~60°
            A.Rotate(limit=[89, 91], p=0.1),      # ~90°
            A.Rotate(limit=[119, 121], p=0.1),    # ~120°
            A.Rotate(limit=[149, 151], p=0.1),    # ~150°
            A.Rotate(limit=[179, 181], p=0.1),    # ~180°
            A.Rotate(limit=[209, 211], p=0.1),    # ~210°
            A.Rotate(limit=[239, 241], p=0.1),    # ~240°
            A.Rotate(limit=[269, 271], p=0.1),    # ~270°
            A.Rotate(limit=[299, 301], p=0.1),    # ~300°
            A.Rotate(limit=[329, 331], p=0.1),    # ~330°
        ], p=0.8),

        # 2. Flipping as mentioned in the paper
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # 3. Subtle intensity adjustments (being very conservative here)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(90, 110), p=0.5)
        ], p=0.3),

        # 4. Subtle noise and blur (to simulate different imaging conditions)
        A.OneOf([
            A.GaussNoise(var_limit=(5, 20), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        ], p=0.2),

        # 5. Minor scaling (to simulate different distances/perspectives)
        A.RandomScale(scale_limit=(-0.1, 0.1), p=0.3),

        # 6. Border handling for rotations
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

# Usage example:
if __name__ == "__main__":
    transform = get_mammography_augmentation()
    
    # Example usage with an image:
    # augmented = transform(image=image)['image']   