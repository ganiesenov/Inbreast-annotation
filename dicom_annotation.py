import os
import json
import pydicom
import pandas as pd
import numpy as np
from pathlib import Path

def clean_value(value):
    """Cleans string value by removing extra spaces and newlines"""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    return value

def get_mass_type(row):
    """Determines if mass is benign or malignant based on BI-RADS and findings"""
    birads = clean_value(row['Bi-Rads'])
    findings = str(clean_value(row.get('Findings Notes (in Portuguese)', ''))).lower()
    
    # Если нет массы, возвращаем None
    if clean_value(row.get('Mass ')) != 'X':
        return None
    
    # Определяем злокачественность
    if isinstance(birads, (int, float)):
        is_malignant = float(birads) >= 4
    elif isinstance(birads, str):
        is_malignant = any(birads.startswith(str(i)) for i in [4, 5, 6])
    else:
        is_malignant = False
    
    # Проверяем описание
    if 'benigno' in findings or 'normal' in findings:
        is_malignant = False
    
    return 'Malignant' if is_malignant else 'Benign'

def create_annotation(row, img_info):
    """Creates annotation based on Excel row data"""
    if not img_info:
        return None
    
    # Get density and mass type
    density = clean_value(row['ACR'])
    mass_type = get_mass_type(row)
    
    # Create category string like "Density1+Benign"
    if mass_type:
        category = f"Density{density}+{mass_type}"
    else:
        category = f"Density{density}"
    
    annotation = {
        'filename': str(int(float(clean_value(row['File Name'])))),
        'image': {
            'width': img_info['width'],
            'height': img_info['height'],
            'pixel_spacing': img_info['spacing']
        },
        'classification': {
            'density': int(density) if density else None,
            'mass_type': mass_type,
            'category': category,
            'birads': clean_value(row['Bi-Rads'])
        },
        'exam': {
            'laterality': clean_value(row['Laterality']),
            'view': clean_value(row['View']),
            'date': str(clean_value(row['Acquisition date']))
        },
        'findings': {
            'mass': clean_value(row.get('Mass ')) == 'X',
            'calcification': clean_value(row.get('Micros ')) == 'X',
            'distortion': clean_value(row.get('Distortion')) == 'X',
            'asymmetry': clean_value(row.get('Asymmetry')) == 'X',
            'description': clean_value(row.get('Findings Notes (in Portuguese)')),
            'lesion_annotation': clean_value(row.get('Lesion Annotation Status'))
        }
    }
    
    return annotation

def main():
    # Setup paths
    current_dir = Path.cwd()
    excel_path = current_dir / 'INbreast.xls'
    dicom_dir = current_dir / 'ALL-IMGS'
    annotations_dir = current_dir / 'annotations'
    
    # Create annotations directory if it doesn't exist
    annotations_dir.mkdir(exist_ok=True)
    
    # Load Excel data
    print("Loading Excel data...")
    df = pd.read_excel(excel_path)
    
    # Process each row
    annotations = []
    categories_count = {}  # Для подсчета количества изображений в каждой категории
    
    print("\nProcessing DICOM files...")
    
    for idx, row in df.iterrows():
        try:
            if pd.isna(row['File Name']):
                continue
                
            filename = str(int(float(clean_value(row['File Name']))))
            dicom_paths = list(dicom_dir.glob(f"*{filename}*.dcm"))
            
            if not dicom_paths:
                print(f"Warning: No DICOM file found for {filename}")
                continue
                
            dicom_path = dicom_paths[0]
            img_info = get_dicom_info(dicom_path)
            
            if img_info:
                annotation = create_annotation(row, img_info)
                if annotation:
                    annotations.append(annotation)
                    # Подсчет категорий
                    category = annotation['classification']['category']
                    categories_count[category] = categories_count.get(category, 0) + 1
                    print(f"Processed image {filename} - {category}")
        
        except Exception as e:
            print(f"Error processing row {idx} (File: {row.get('File Name')}): {str(e)}")
            continue
    
    # Save annotations
    output_path = annotations_dir / 'all_annotations.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\nDataset statistics:")
    print(f"Total images processed: {len(annotations)}")
    print("\nImages per category:")
    for category, count in sorted(categories_count.items()):
        print(f"{category}: {count}")
    
    print(f"\nDone! Annotations saved to {output_path}")

if __name__ == '__main__':
    main()