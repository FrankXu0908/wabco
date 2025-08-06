from pathlib import Path
import shutil
import random

def split_dataset_detection(dataset_dir="samples", train_ratio=0.8):
    """
    Split the dataset into training and validation sets for both images and labels.
    """
    # Base path
    base_dir = Path(dataset_dir)
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    train_ratio = train_ratio
    print(images_dir, labels_dir)
    
    # Output subdirs
    images_train_dir = images_dir / "train"
    images_val_dir = images_dir / "val"
    labels_train_dir = labels_dir / "train"
    labels_val_dir = labels_dir / "val"

    # Create output subdirs
    images_train_dir.mkdir(exist_ok=True)
    images_val_dir.mkdir(exist_ok=True)
    labels_train_dir.mkdir(exist_ok=True)
    labels_val_dir.mkdir(exist_ok=True)

    # Get sorted lists of all image and label files
    image_files = sorted(images_dir.glob("*.jpg"), key=lambda x: x.stem)
    labels_files = sorted(labels_dir.glob("*.txt"), key=lambda x: x.stem)

    # Match only files with both image and label
    matched_pairs = [
        (img, labels_dir / (img.stem + ".txt"))
        for img in image_files
        if (labels_dir / (img.stem + ".txt")).exists()
    ]

    # Shuffle and split
    random.shuffle(matched_pairs)
    split_idx = int(len(matched_pairs) * train_ratio)
    print(split_idx)
    train_pairs = matched_pairs[:split_idx]
    val_pairs = matched_pairs[split_idx:]

    # Move train images and labels
    for img_path, lbl_path in train_pairs:
        shutil.move(str(img_path), str(images_train_dir / img_path.name))
        shutil.move(str(lbl_path), str(labels_train_dir / lbl_path.name))

    # Move val images and labels
    for img_path, lbl_path in val_pairs:
        shutil.move(str(img_path), str(images_val_dir / img_path.name))
        shutil.move(str(lbl_path), str(labels_val_dir / lbl_path.name))

def split_dataset_classification(dataset_dir="dataset/biclassifier_v1", train_ratio=0.8):
    """
    Split the dataset into training and validation sets for both images and labels.
    """
    # Base path
    base_dir = Path(dataset_dir)
    images_dir_ng = base_dir / "NG_cropped"
    images_dir_ok = base_dir / "OK_cropped"
    train_ratio = train_ratio
    print(images_dir_ng, images_dir_ok)

    # Output subdirs
    images_train_dir_ng = images_dir_ng / "train"
    images_val_dir_ng = images_dir_ng / "val"
    images_train_dir_ok = images_dir_ok / "train"
    images_val_dir_ok = images_dir_ok / "val"

    # Create output subdirs
    images_train_dir_ng.mkdir(exist_ok=True)
    images_val_dir_ng.mkdir(exist_ok=True)
    images_train_dir_ok.mkdir(exist_ok=True)
    images_val_dir_ok.mkdir(exist_ok=True)

    # Get sorted lists of all image and label files
    image_files_ng = sorted(images_dir_ng.glob("*.bmp"), key=lambda x: x.stem)
    image_files_ok = sorted(images_dir_ok.glob("*.bmp"), key=lambda x: x.stem)

    # Shuffle and split
    random.shuffle(image_files_ng)
    split_idx_ng = int(len(image_files_ng) * train_ratio)
    print(split_idx_ng)

    random.shuffle(image_files_ok)
    split_idx_ok = int(len(image_files_ok) * train_ratio)
    print(split_idx_ok)
    train_ng = image_files_ng[:split_idx_ng]
    val_ng = image_files_ng[split_idx_ng:]
    train_ok = image_files_ok[:split_idx_ok]
    val_ok = image_files_ok[split_idx_ok:]

    # Move train images and labels
    for img_path in train_ng:
        shutil.move(str(img_path), str(images_train_dir_ng / img_path.name))
    for img_path in train_ok:
        shutil.move(str(img_path), str(images_train_dir_ok / img_path.name))

    # Move val images and labels
    for img_path in val_ng:
        shutil.move(str(img_path), str(images_val_dir_ng / img_path.name))
    for img_path in val_ok:
        shutil.move(str(img_path), str(images_val_dir_ok / img_path.name))

split_dataset_detection()