import os
from pathlib import Path

def count_files_in_augmented_dataset(base_path="augmented_dataset"):
    categories = [
        "Density1+Benign",
        "Density1+Malignant", 
        "Density2+Benign",
        "Density2+Malignant",
        "Density3+Benign",
        "Density3+Malignant",
        "Density4+Benign",
        "Density4+Malignant"
    ]
    
    total_files = 0
    print("\nFiles distribution:")
    print("-" * 40)
    
    for category in categories:
        path = Path(base_path) / category
        if path.exists():
            files = list(path.glob("*"))
            num_files = len(files)
            total_files += num_files
            print(f"{category}: {num_files:,} files")
    
    print("-" * 40)
    print(f"Total files: {total_files:,}")

if __name__ == "__main__":
    count_files_in_augmented_dataset()