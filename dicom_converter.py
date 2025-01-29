import os
import json
import numpy as np
import pydicom
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Ожидаемое количество изображений в каждой категории
EXPECTED_COUNTS = {
    'Density1+Benign': 12,
    'Density1+Malignant': 30,
    'Density2+Benign': 4,    # Исправлено с 6 на 4
    'Density2+Malignant': 32,
    'Density3+Benign': 13,
    'Density3+Malignant': 8,
    'Density4+Benign': 6,
    'Density4+Malignant': 1
}

def normalize_dicom(dicom_data):
    """Normalize DICOM pixel array to 0-255 range"""
    pixel_array = dicom_data.pixel_array
    if pixel_array.max() != pixel_array.min():
        normalized = ((pixel_array - pixel_array.min()) * 255.0 / (pixel_array.max() - pixel_array.min()))
    else:
        normalized = pixel_array * 0
    return normalized.astype('uint8')

def convert_dicom_to_png(dicom_path, output_path, file_format='PNG'):
    """Convert DICOM file to PNG/JPG"""
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        img_array = normalize_dicom(dicom_data)
        image = Image.fromarray(img_array)
        image.save(output_path, format=file_format)
        return True
    except Exception as e:
        print(f"Error converting {dicom_path}: {str(e)}")
        return False

def load_annotations(annotations_path):
    """Load and filter annotations to include only mass cases"""
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Группируем аннотации по категориям
    category_annotations = defaultdict(list)
    for ann in annotations:
        if 'classification' in ann and 'category' in ann['classification']:
            category = ann['classification']['category']
            if category in EXPECTED_COUNTS:  # только нужные категории
                category_annotations[category].append(ann)
    
    return category_annotations

def select_cases(category_annotations):
    """Select cases to match expected counts"""
    selected_annotations = []
    
    for category, expected_count in EXPECTED_COUNTS.items():
        available = category_annotations[category]
        if len(available) < expected_count:
            print(f"Warning: Not enough cases for {category}. Expected {expected_count}, found {len(available)}")
            selected = available
        elif len(available) > expected_count:
            print(f"Note: Selecting {expected_count} cases from {len(available)} available for {category}")
            # Используем seed для воспроизводимости
            np.random.seed(42)
            selected = np.random.choice(available, expected_count, replace=False).tolist()
        else:
            selected = available
            
        selected_annotations.extend(selected)
    
    return selected_annotations

def main():
    current_dir = Path.cwd()
    dicom_dir = current_dir / 'ALL-IMGS'
    annotations_path = current_dir / 'annotations' / 'all_annotations.json'
    output_base = current_dir / 'mass_images'
    
    # Загружаем и фильтруем аннотации
    print("Loading annotations...")
    category_annotations = load_annotations(annotations_path)
    
    # Выбираем нужное количество случаев
    selected_annotations = select_cases(category_annotations)
    
    # Создаем директории для каждой категории
    for category in EXPECTED_COUNTS:
        (output_base / category).mkdir(parents=True, exist_ok=True)
    
    # Конвертируем отобранные случаи
    print("\nConverting selected DICOM files...")
    successful = defaultdict(int)
    failed = []
    
    for ann in tqdm(selected_annotations):
        try:
            filename = ann['filename']
            category = ann['classification']['category']
            
            # Ищем DICOM файл
            dicom_paths = list(dicom_dir.glob(f"*{filename}*.dcm"))
            if not dicom_paths:
                print(f"Warning: No DICOM file found for {filename}")
                failed.append((filename, "File not found"))
                continue
            
            # Конвертируем
            output_path = output_base / category / f"{filename}.png"
            if convert_dicom_to_png(dicom_paths[0], output_path):
                successful[category] += 1
            else:
                failed.append((filename, "Conversion failed"))
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            failed.append((filename, str(e)))
    
    # Выводим итоговую статистику
    print("\nConversion completed!")
    print("\nResults by category:")
    print("╔════════════════════╦═══════════╦════════════╦════════════╗")
    print("║      Category      ║  Expected ║  Converted ║ Difference ║")
    print("╠════════════════════╬═══════════╬════════════╬════════════╣")
    
    total_expected = 0
    total_converted = 0
    
    for category in EXPECTED_COUNTS:
        expected = EXPECTED_COUNTS[category]
        converted = successful[category]
        diff = converted - expected
        total_expected += expected
        total_converted += converted
        
        print(f"║ {category:<16} ║ {expected:^9} ║ {converted:^10} ║ {diff:^10} ║")
    
    print("╠════════════════════╬═══════════╬════════════╬════════════╣")
    print(f"║ Total             ║ {total_expected:^9} ║ {total_converted:^10} ║ {total_converted-total_expected:^10} ║")
    print("╚════════════════════╩═══════════╩════════════╩════════════╝")
    
    if failed:
        print("\nFailed conversions:")
        for filename, reason in failed:
            print(f"- {filename}: {reason}")
    
    print(f"\nImages are saved in: {output_base}")

if __name__ == '__main__':
    main()