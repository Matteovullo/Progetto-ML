import os
import cv2
import numpy as np
from PIL import Image
import random
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

gt_file = "/Users/matteovullo/Downloads/TrainIJCNN2013-3/gt.txt"
images_dir = "/Users/matteovullo/Downloads/TrainIJCNN2013-3"
output_dir = "gtsdb_yolo_format"

CLASS_NAMES_IN_GTSDB_ORDER = [
    'speed-limit-20', 'speed-limit-30', 'speed-limit-50',
    'speed-limit-60', 'speed-limit-70', 'speed-limit-80',
    'restriction-ends-80', 'speed-limit-100', 'speed-limit-120',
    'no-overtaking', 'no-overtaking-trucks', 
    'priority-at-next-intersection', 'priority-road', 'give-way',
    'stop', 'no-traffic-both-ways', 'no-trucks', 'no-entry',
    'danger', 'bend-left', 'bend-right', 'bend', 'uneven-road',
    'slippery-road', 'road-narrows', 'construction',
    'traffic-signal', 'pedestrian-crossing', 'school-crossing',
    'cycles-crossing', 'snow', 'animals', 'restriction-ends',
    'go-right', 'go-left', 'go-straight', 'go-right-or-straight',
    'go-left-or-straight', 'keep-right', 'keep-left',
    'roundabout', 'restriction-ends-overtaking',
    'restriction-ends-overtaking-trucks'
]

gtsdb_to_yolo_class_id = {i: i for i in range(43)}

os.makedirs(output_dir, exist_ok=True)
for split in ["train", "val"]:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

all_image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg'))]
print(f"Trovate {len(all_image_files)} immagini totali nella directory")

annotations_by_image = {}
if os.path.exists(gt_file):
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Lettura {len(lines)} linee dal file gt.txt")
    
    for line_num, line in enumerate(lines, 1):
        parts = line.strip().split(';')
        if len(parts) != 6:
            continue
        img_name = parts[0].strip()
        if img_name not in annotations_by_image:
            annotations_by_image[img_name] = []
        annotations_by_image[img_name].append((line, line_num))
    
    print(f"Trovate {len(annotations_by_image)} immagini con annotazioni nel file gt.txt")
    
else:
    print(f"File gt.txt non trovato: {gt_file}")

images_with_annotations = set(annotations_by_image.keys())
all_images_set = set(all_image_files)

valid_images = []
for img_name in all_images_set:
    img_path = os.path.join(images_dir, img_name)
    if os.path.exists(img_path):
        valid_images.append(img_name)

print(f"Immagini con annotazioni: {len(images_with_annotations)}")
print(f"Immagini senza annotazioni: {len(all_images_set - images_with_annotations)}")
print(f"Immagini valide trovate: {len(valid_images)}")

print("\n=== FASE 1: Verifica immagini processabili ===")
processable_images = []
failed_precheck = []

for img_name in valid_images:
    img_path = os.path.join(images_dir, img_name)
    try:
        img = Image.open(img_path)
        img.verify()
        processable_images.append(img_name)
    except Exception as e:
        print(f"Immagine non processabile: {img_name} -> {e}")
        failed_precheck.append(img_name)

print(f"Immagini processabili: {len(processable_images)}")
print(f"Immagini scartate in pre-check: {len(failed_precheck)}")

random.seed(42)
random.shuffle(processable_images)
n_train = int(0.8 * len(processable_images))
train_images = processable_images[:n_train]
val_images = processable_images[n_train:]

print(f"\n=== Split definitivo sulle immagini processabili ===")
print(f"Train: {len(train_images)} immagini")
print(f"Val:   {len(val_images)} immagini")
print(f"Rapporto: {len(train_images)/len(processable_images)*100:.1f}% / {len(val_images)/len(processable_images)*100:.1f}%")

successful_train = []
successful_val = []
failed_processing = []
total_annotations_written = 0

print(f"\n=== FASE 2: Processing immagini ===")
for i, img_name in enumerate(train_images):
    split = "train"
    
    new_img_name = img_name.replace('.ppm', '.jpg')
    img_out_path = os.path.join(output_dir, split, "images", new_img_name)
    label_out_path = os.path.join(output_dir, split, "labels", new_img_name.replace('.jpg', '.txt'))

    img_path = os.path.join(images_dir, img_name)
    
    try:
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        h, w = img.shape[:2]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        success = cv2.imwrite(img_out_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise Exception(f"Impossibile salvare l'immagine")
        
        annotations_written = 0
        with open(label_out_path, 'w') as label_file:
            if img_name in annotations_by_image:
                for line, line_num in annotations_by_image[img_name]:
                    parts = line.strip().split(';')
                    if len(parts) != 6:
                        continue
                        
                    try:
                        left = int(parts[1])
                        top = int(parts[2])
                        right = int(parts[3])
                        bottom = int(parts[4])
                        gtsdb_class_id = int(parts[5])
                    except ValueError:
                        continue

                    yolo_class_id = gtsdb_to_yolo_class_id.get(gtsdb_class_id)
                    if yolo_class_id is None:
                        continue

                    x_center = (left + right) / 2 / w
                    y_center = (top + bottom) / 2 / h
                    width = (right - left) / w
                    height = (bottom - top) / h

                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                        label_file.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        annotations_written += 1
                
                total_annotations_written += annotations_written
        
        successful_train.append(img_name)
        if annotations_written > 0:
            print(f"TRAIN: {img_name} - {annotations_written} annotazioni")
            
    except Exception as e:
        print(f"ERRORE TRAIN: {img_name} -> {e}")
        failed_processing.append(img_name)
    
    if (i + 1) % 20 == 0: 
        print(f"Training: processate {i + 1}/{len(train_images)} immagini")

for i, img_name in enumerate(val_images):
    split = "val"
    
    new_img_name = img_name.replace('.ppm', '.jpg')
    img_out_path = os.path.join(output_dir, split, "images", new_img_name)
    label_out_path = os.path.join(output_dir, split, "labels", new_img_name.replace('.jpg', '.txt'))

    img_path = os.path.join(images_dir, img_name)
    
    try:
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        h, w = img.shape[:2]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        success = cv2.imwrite(img_out_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise Exception(f"Impossibile salvare l'immagine")

        annotations_written = 0
        with open(label_out_path, 'w') as label_file:
            if img_name in annotations_by_image:
                for line, line_num in annotations_by_image[img_name]:
                    parts = line.strip().split(';')
                    if len(parts) != 6:
                        continue
                        
                    try:
                        left = int(parts[1])
                        top = int(parts[2])
                        right = int(parts[3])
                        bottom = int(parts[4])
                        gtsdb_class_id = int(parts[5])
                    except ValueError:
                        continue

                    yolo_class_id = gtsdb_to_yolo_class_id.get(gtsdb_class_id)
                    if yolo_class_id is None:
                        continue

                    x_center = (left + right) / 2 / w
                    y_center = (top + bottom) / 2 / h
                    width = (right - left) / w
                    height = (bottom - top) / h

                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1:
                        label_file.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        annotations_written += 1
                
                total_annotations_written += annotations_written
        
        successful_val.append(img_name)
        if annotations_written > 0:
            print(f"VAL: {img_name} - {annotations_written} annotazioni")
            
    except Exception as e:
        print(f"ERRORE VAL: {img_name} -> {e}")
        failed_processing.append(img_name)
    
    if (i + 1) % 10 == 0: 
        print(f"Validation: processate {i + 1}/{len(val_images)} immagini")

train_images_saved = len([f for f in os.listdir(os.path.join(output_dir, "train", "images")) if f.lower().endswith('.jpg')])
val_images_saved = len([f for f in os.listdir(os.path.join(output_dir, "val", "images")) if f.lower().endswith('.jpg')])

print(f"\n=== RIEPILOGO FINALE ===")
print(f"Immagini processate con successo: {len(successful_train) + len(successful_val)}")
print(f"  - Training: {len(successful_train)} immagini")
print(f"  - Validation: {len(successful_val)} immagini")
print(f"Immagini fallite nel processing: {len(failed_processing)}")
print(f"Immagini scartate in pre-check: {len(failed_precheck)}")
print(f"Annotazioni totali scritte: {total_annotations_written}")

total_successful = len(successful_train) + len(successful_val)
if total_successful > 0:
    train_percentage = len(successful_train) / total_successful * 100
    val_percentage = len(successful_val) / total_successful * 100
    print(f"\n=== VERIFICA SPLIT ===")
    print(f"Rapporto finale Train/Val: {train_percentage:.1f}% / {val_percentage:.1f}%")
    
    if abs(train_percentage - 80.0) < 1: 
        print("Split bilanciato correttamente!")
    else:
        print("Split leggermente sbilanciato")

train_labels_non_empty = 0
val_labels_non_empty = 0

for label_file in os.listdir(os.path.join(output_dir, "train", "labels")):
    if label_file.endswith('.txt'):
        file_path = os.path.join(output_dir, "train", "labels", label_file)
        if os.path.getsize(file_path) > 0:
            train_labels_non_empty += 1

for label_file in os.listdir(os.path.join(output_dir, "val", "labels")):
    if label_file.endswith('.txt'):
        file_path = os.path.join(output_dir, "val", "labels", label_file)
        if os.path.getsize(file_path) > 0:
            val_labels_non_empty += 1

print(f"\n=== FILE SALVATI ===")
print(f"Train: {train_images_saved} immagini, {train_labels_non_empty} label con annotazioni")
print(f"Val:   {val_images_saved} immagini, {val_labels_non_empty} label con annotazioni")

data_yaml_content = f"""
train: {output_dir}/train/images
val: {output_dir}/val/images

nc: {len(CLASS_NAMES_IN_GTSDB_ORDER)}
names: {CLASS_NAMES_IN_GTSDB_ORDER}
"""

with open(os.path.join(output_dir, "data.yaml"), 'w') as f:
    f.write(data_yaml_content.strip())

print("\nFile data.yaml creato con successo!")
print(f"Dataset YOLO pronto in: {output_dir}")