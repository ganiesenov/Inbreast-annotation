import os
import json
import pydicom
import pandas as pd
import numpy as np
from pathlib import Path

def get_dicom_info(dicom_path):
    """Extracts basic info from DICOM file"""
    try:
        dcm = pydicom.dcmread(dicom_path)
        return {
            'width': dcm.Columns,
            'height': dcm.Rows,
            'spacing': getattr(dcm, 'PixelSpacing', [1, 1])
        }
    except Exception as e:
        print(f"Error reading DICOM {dicom_path}: {e}")
        return None

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
    if not density or not str(density).isdigit():
        print(f"Warning: Invalid density value for file {row['File Name']}: {density}")
        return None
        
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
            'density': int(density),
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
    print(f"Loaded {len(df)} rows from Excel")
    
    # Process each row
    annotations = []
    categories_count = {}  # Для подсчета количества изображений в каждой категории
    errors = []  # Для сбора информации об ошибках
    
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
            error_msg = f"Error processing row {idx} (File: {row.get('File Name')}): {str(e)}"
            print(error_msg)
            errors.append(error_msg)
            continue
    
    # Save annotations
    output_path = annotations_dir / 'all_annotations.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\nDataset statistics:")
    print(f"Total images processed: {len(annotations)}")
    print(f"Failed to process: {len(errors)} images")
    
    print("\nImages per category:")
    for category, count in sorted(categories_count.items()):
        print(f"{category}: {count}")
    
    # Save error log if there were any errors
    if errors:
        error_path = annotations_dir / 'processing_errors.txt'
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(errors))
        print(f"\nErrors have been saved to {error_path}")
    
    print(f"\nDone! Annotations saved to {output_path}")

if __name__ == '__main__':
    main()