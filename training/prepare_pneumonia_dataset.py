"""
Script to prepare pneumonia dataset from Kaggle or other sources.

This script organizes chest X-ray images into train/val/test splits.
"""

import os
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

def organize_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Organize dataset into train/val/test splits.
    
    Args:
        source_dir: Source directory containing images
        output_dir: Output directory for organized dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Create output directory structure
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        for class_name in ['pneumonia', 'normal']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Find all images
    source_path = Path(source_dir)
    
    # Common dataset structures
    possible_structures = [
        # Structure 1: train/test with NORMAL and PNEUMONIA folders
        (source_path / 'train' / 'NORMAL', source_path / 'train' / 'PNEUMONIA'),
        (source_path / 'test' / 'NORMAL', source_path / 'test' / 'PNEUMONIA'),
        # Structure 2: Direct NORMAL and PNEUMONIA folders
        (source_path / 'NORMAL', source_path / 'PNEUMONIA'),
        # Structure 3: normal and pneumonia (lowercase)
        (source_path / 'normal', source_path / 'pneumonia'),
    ]
    
    normal_images = []
    pneumonia_images = []
    
    # Try to find images in different structures
    for normal_dir, pneumonia_dir in possible_structures:
        if normal_dir.exists():
            normal_images.extend(list(normal_dir.glob('*.jpeg')) + list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.png')))
        if pneumonia_dir.exists():
            pneumonia_images.extend(list(pneumonia_dir.glob('*.jpeg')) + list(pneumonia_dir.glob('*.jpg')) + list(pneumonia_dir.glob('*.png')))
    
    if not normal_images and not pneumonia_images:
        # Try recursive search
        print("Searching recursively for images...")
        all_images = list(source_path.rglob('*.jpeg')) + list(source_path.rglob('*.jpg')) + list(source_path.rglob('*.png'))
        
        # Try to classify by folder name
        for img_path in all_images:
            parent_name = img_path.parent.name.lower()
            if 'normal' in parent_name or 'nrml' in parent_name:
                normal_images.append(img_path)
            elif 'pneumonia' in parent_name or 'pneum' in parent_name:
                pneumonia_images.append(img_path)
    
    print(f"Found {len(normal_images)} normal images")
    print(f"Found {len(pneumonia_images)} pneumonia images")
    
    if not normal_images or not pneumonia_images:
        print("ERROR: Could not find images. Please check your dataset structure.")
        print("\nExpected structure:")
        print("  source_dir/")
        print("    NORMAL/ or normal/")
        print("      *.jpg")
        print("    PNEUMONIA/ or pneumonia/")
        print("      *.jpg")
        return
    
    # Shuffle images
    random.shuffle(normal_images)
    random.shuffle(pneumonia_images)
    
    # Split each class
    def split_images(images, train_r, val_r, test_r):
        n = len(images)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        
        train = images[:n_train]
        val = images[n_train:n_train+n_val]
        test = images[n_train+n_val:]
        
        return train, val, test
    
    normal_train, normal_val, normal_test = split_images(normal_images, train_ratio, val_ratio, test_ratio)
    pneumonia_train, pneumonia_val, pneumonia_test = split_images(pneumonia_images, train_ratio, val_ratio, test_ratio)
    
    # Copy images to output directories
    def copy_images(images, dest_dir, class_name):
        dest_path = output_path / dest_dir / class_name
        for img_path in images:
            shutil.copy2(img_path, dest_path / img_path.name)
    
    print("\nCopying images...")
    copy_images(normal_train, 'train', 'normal')
    copy_images(normal_val, 'val', 'normal')
    copy_images(normal_test, 'test', 'normal')
    copy_images(pneumonia_train, 'train', 'pneumonia')
    copy_images(pneumonia_val, 'val', 'pneumonia')
    copy_images(pneumonia_test, 'test', 'pneumonia')
    
    print("\nDataset organization complete!")
    print(f"\nDataset structure:")
    print(f"  Train: {len(normal_train)} normal, {len(pneumonia_train)} pneumonia")
    print(f"  Val: {len(normal_val)} normal, {len(pneumonia_val)} pneumonia")
    print(f"  Test: {len(normal_test)} normal, {len(pneumonia_test)} pneumonia")
    print(f"\nOutput directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Prepare pneumonia dataset')
    parser.add_argument('--source_dir', type=str, required=True, help='Source directory with images')
    parser.add_argument('--output_dir', type=str, default='./data/pneumonia_dataset', help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    
    args = parser.parse_args()
    
    organize_dataset(
        args.source_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )

if __name__ == '__main__':
    main()

