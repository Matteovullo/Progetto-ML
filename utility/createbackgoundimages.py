import os
import shutil
import random

IMAGE_DIR = "/Users/matteovullo/Downloads/Traffic_Signs_Detection_ML.v3i.yolov8-obb/test/images"      
LABEL_DIR = "/Users/matteovullo/Downloads/Traffic_Signs_Detection_ML.v3i.yolov8-obb/test/labels"     
OUTPUT_IMAGE_DIR = "dataset/dataset_background_test/images_bg"
OUTPUT_LABEL_DIR = "dataset/dataset_background_test/labels_bg"
NUM_BACKGROUND_IMAGES = 50   

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(image_extensions)]

if len(image_files) < NUM_BACKGROUND_IMAGES:
    print(f"Hai solo {len(image_files)} immagini, ma ne vuoi {NUM_BACKGROUND_IMAGES}. Ne prenderò tutte.")
    selected_images = image_files
else:
    selected_images = random.sample(image_files, NUM_BACKGROUND_IMAGES)

for img_file in selected_images:
    img_path = os.path.join(IMAGE_DIR, img_file)
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_file)

    new_img_path = os.path.join(OUTPUT_IMAGE_DIR, img_file)
    new_label_path = os.path.join(OUTPUT_LABEL_DIR, label_file)

    shutil.copy(img_path, new_img_path)

    with open(new_label_path, 'w') as f:
        pass  

    print(f"Background creato: {img_file}")

print(f"\nFatto! Create {len(selected_images)} immagini di background.")
print(f"Immagini salvate in: {OUTPUT_IMAGE_DIR}")
print(f"Etichette vuote salvate in: {OUTPUT_LABEL_DIR}")