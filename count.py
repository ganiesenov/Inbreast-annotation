import json
from collections import Counter
from pathlib import Path

def print_comparison_table():
    # Ожидаемые значения из таблицы
    expected_counts = {
        'Density1+Benign': 12,
        'Density1+Malignant': 30,
        'Density2+Benign': 4,
        'Density2+Malignant': 32,
        'Density3+Benign': 13,
        'Density3+Malignant': 8,
        'Density4+Benign': 6,
        'Density4+Malignant': 1
    }

    # Загружаем наши аннотации
    annotations_path = Path('annotations/all_annotations.json')
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # Подсчитываем категории в наших аннотациях
    actual_counts = Counter()
    for ann in annotations:
        if 'classification' in ann and 'category' in ann['classification']:
            category = ann['classification']['category']
            if '+' in category:  # Учитываем только категории с массами
                actual_counts[category] += 1

    # Печатаем сравнительную таблицу
    print("\nComparison with expected distribution:")
    print("╔════════════════════╦═══════════╦════════════╦════════════╗")
    print("║      Category      ║  Expected ║   Actual   ║ Difference ║")
    print("╠════════════════════╬═══════════╬════════════╬════════════╣")
    
    total_expected = 0
    total_actual = 0
    
    for category in expected_counts:
        expected = expected_counts[category]
        actual = actual_counts[category]
        diff = actual - expected
        
        total_expected += expected
        total_actual += actual
        
        print(f"║ {category:<16} ║ {expected:^9} ║ {actual:^10} ║ {diff:^10} ║")
        
    print("╠════════════════════╬═══════════╬════════════╬════════════╣")
    print(f"║ Total             ║ {total_expected:^9} ║ {total_actual:^10} ║ {total_actual-total_expected:^10} ║")
    print("╚════════════════════╩═══════════╩════════════╩════════════╝")

    # Проверяем различия
    if total_expected != total_actual:
        print(f"\nWarning: Total number of samples differs!")
        print(f"Expected: {total_expected}, Found: {total_actual}")

    # Находим пропущенные или лишние категории
    missing_categories = set(expected_counts.keys()) - set(actual_counts.keys())
    extra_categories = set(actual_counts.keys()) - set(expected_counts.keys())
    
    if missing_categories:
        print("\nMissing categories:", missing_categories)
    if extra_categories:
        print("\nUnexpected categories:", extra_categories)

    # Подробная статистика по всем найденным изображениям
    print("\nDetailed statistics:")
    print(f"Total images in annotations: {len(annotations)}")
    
    # Распределение по плотности
    density_counts = Counter()
    for ann in annotations:
        if 'classification' in ann and 'density' in ann['classification']:
            density_counts[f"Density {ann['classification']['density']}"] += 1
    
    print("\nDistribution by density:")
    for density, count in sorted(density_counts.items()):
        print(f"{density}: {count}")

if __name__ == '__main__':
    print_comparison_table()