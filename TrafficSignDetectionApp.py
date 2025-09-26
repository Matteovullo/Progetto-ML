import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from collections import Counter
from ultralytics import YOLO
import json
import os
import numpy as np

img = None
img_path = None
model = None
empty_img = None 

MAX_IMAGE_SIZE = (400, 400)

CLASS_NAMES = {
    0: "forb_overtake",
    1: "forb_overtake_trucks", 
    2: "forb_speed_over_10",
    3: "forb_speed_over_100",
    4: "forb_speed_over_120",
    5: "forb_speed_over_20",
    6: "forb_speed_over_30",
    7: "forb_speed_over_40",
    8: "forb_speed_over_50",
    9: "forb_speed_over_60",
    10: "forb_speed_over_70",
    11: "forb_speed_over_80",
    12: "forb_stopping",
    13: "forb_waiting",
    14: "info_crosswalk",
    15: "prio_give_way",
    16: "prio_stop",
    17: "warn_crosswalk",
    18: "warn_double_curve",
    19: "warn_left_curve",
    20: "warn_right_curve"
}

def load_image():
    global img, img_path
    remove_prediction()
    img_path = filedialog.askopenfilename(
        title="Seleziona un'immagine",
        filetypes=[("Immagini", "*.jpg *.jpeg *.png *.bmp")]
    )
    if img_path:
        img = cv2.imread(img_path)
        if img is None:
            messagebox.showerror("Errore", "Impossibile caricare l'immagine.")
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = resize_image(img_pil, MAX_IMAGE_SIZE)
        img_tk = ImageTk.PhotoImage(img_pil)
        img_panel_left.config(image=img_tk)
        img_panel_left.image = img_tk
        title_left.grid()
        
        remove_btn.config(state=tk.NORMAL)
        
        load_ground_truth(img_path)
        
        root.update_idletasks()

def resize_image(image, max_size):
    image = image.copy()
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def load_ground_truth(image_path):
    base_name = os.path.splitext(image_path)[0]
    ground_truth_img = None
    
    yolo_label_path = base_name + '.txt'
    if os.path.exists(yolo_label_path):
        ground_truth_img = create_ground_truth_from_polygon(img, yolo_label_path)
    
    json_label_path = base_name + '.json'
    if os.path.exists(json_label_path) and ground_truth_img is None:
        ground_truth_img = create_ground_truth_from_json(img, json_label_path)
    
    if ground_truth_img is None:
        img_panel_center.config(image=empty_img)
        img_panel_center.image = empty_img
        title_center.grid_remove()
        messagebox.showinfo("Info", "Nessuna annotazione ground truth trovata per questa immagine.")
    else:
        img_pil = Image.fromarray(ground_truth_img)
        img_pil = resize_image(img_pil, MAX_IMAGE_SIZE)
        img_tk = ImageTk.PhotoImage(img_pil)
        img_panel_center.config(image=img_tk)
        img_panel_center.image = img_tk
        title_center.grid()

def create_ground_truth_from_polygon(image, label_path):
    img_height, img_width = image.shape[:2]
    img_gt = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
            if not lines:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            overlay = img_gt.copy()
                
            for i, line in enumerate(lines):
                data = line.strip().split()
                
                if len(data) == 9:
                    class_id = int(data[0])
                    coords = list(map(float, data[1:9]))
                    points = []
                    for j in range(0, 8, 2):
                        x = int(coords[j] * img_width)
                        y = int(coords[j+1] * img_height)
                        points.append([x, y])
                    points = np.array(points, dtype=np.int32)
                    
                    cv2.fillPoly(overlay, [points], color=(0, 255, 0))
                    
                    class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
                    
                    min_y = points[:, 1].min()
                    candidates = points[points[:, 1] == min_y]
                    min_x = candidates[:, 0].min()
                    
                    text = f"GT: {class_name}"
                    font_scale = max(0.5, min(1.0, img_width / 800))
                    thickness = 2
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    
                    pad = 6
                    cv2.rectangle(img_gt, 
                                (min_x - pad, min_y - text_h - pad), 
                                (min_x + text_w + pad, min_y + pad), 
                                (0, 100, 0), -1)
                    cv2.putText(img_gt, text, (min_x, min_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
            
            alpha = 0.2
            img_gt = cv2.addWeighted(overlay, alpha, img_gt, 1 - alpha, 0)
            
            for i, line in enumerate(lines):
                data = line.strip().split()
                if len(data) == 9:
                    coords = list(map(float, data[1:9]))
                    points = []
                    for j in range(0, 8, 2):
                        x = int(coords[j] * img_width)
                        y = int(coords[j+1] * img_height)
                        points.append([x, y])
                    points = np.array(points, dtype=np.int32)
                    cv2.polylines(img_gt, [points], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    
    except Exception as e:
        print(f"Errore nel caricamento ground truth poligonale: {e}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return img_gt

def create_ground_truth_from_json(image, json_path):
    img_gt = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    img_height, img_width = image.shape[:2]
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        image_name = os.path.basename(img_path)
        image_info = None
        for img_info in data.get('images', []):
            if img_info['file_name'] == image_name:
                image_info = img_info
                break
        
        if image_info:
            image_id = image_info['id']
            annotations = [ann for ann in data.get('annotations', []) 
                         if ann['image_id'] == image_id]
            
            for i, ann in enumerate(annotations):
                bbox = ann['bbox']
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[0] + bbox[2])
                y2 = int(bbox[1] + bbox[3])
                class_id = ann['category_id']
                class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
                
                overlay = img_gt.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                img_gt = cv2.addWeighted(overlay, 0.2, img_gt, 0.8, 0)
                cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                text = f"GT: {class_name}"
                font_scale = max(0.5, min(0.8, img_width / 1000))
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                pad = 6
                cv2.rectangle(img_gt, (x1 - pad, y1 - text_h - pad), (x1 + text_w + pad, y1 + pad), (0, 100, 0), -1)
                cv2.putText(img_gt, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
                
    except Exception as e:
        print(f"Errore nel caricamento ground truth JSON: {e}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return img_gt

def remove_prediction():
    title_right.grid_remove()
    img_panel_right.config(image=empty_img)
    img_panel_right.image = empty_img
    root.update_idletasks()

def remove_image():
    global img, img_path
    
    img = None
    img_path = None
    
    title_left.grid_remove()
    title_center.grid_remove()
    title_right.grid_remove()
    
    img_panel_left.config(image=empty_img)
    img_panel_center.config(image=empty_img)
    img_panel_right.config(image=empty_img)
    img_panel_left.image = empty_img
    img_panel_center.image = empty_img
    img_panel_right.image = empty_img
    
    remove_btn.config(state=tk.DISABLED)

    root.update_idletasks()

def load_model(event):
    global model
    model_name = selected_model.get()
    model_paths = {
        "YOLOv10n": "./TrainYolov10n/runs/detect/train/weights/best.pt",
        "YOLOv10s": "./TrainYolov10n/runs/detect/train2/weights/best.pt",
        "YOLOv10m": "./TrainYolov10n/runs/detect/train3/weights/best.pt"
    }
    try:
        model = YOLO(model_paths[model_name])
        messagebox.showinfo("Modello caricato", f"{model_name} caricato con successo!")
    except Exception as e:
        messagebox.showerror("Errore", f"Impossibile caricare il modello: {str(e)}")

def predict():
    global img, img_path
    if img is None:
        messagebox.showerror("Errore", "Carica prima un'immagine!")
        return
    if model is None:
        messagebox.showerror("Errore", "Seleziona un modello!")
        return

    results = model(img_path)

    for result in results:
        image = result.plot() 

    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_pil = resize_image(img_pil, MAX_IMAGE_SIZE)
    img_tk = ImageTk.PhotoImage(img_pil)
    img_panel_right.config(image=img_tk)
    img_panel_right.image = img_tk
    title_right.grid()

    root.update_idletasks()

def count_ground_truth_objects():
    if not img_path:
        return 0
    
    base_name = os.path.splitext(img_path)[0]
    count = 0
    
    yolo_label_path = base_name + '.txt'
    if os.path.exists(yolo_label_path):
        try:
            with open(yolo_label_path, 'r') as f:
                count = len([line for line in f if line.strip()])
        except:
            pass
    
    return count

root = tk.Tk()
root.title("Rilevatore di Segnali Stradali YOLOv10")
root.geometry("1300x750") 

empty_pil = Image.new("RGB", (400, 300), color="lightgray")
empty_img = ImageTk.PhotoImage(empty_pil)

title_app = tk.Label(root, text="Rilevatore di Segnali Stradali", 
                    font=("Helvetica", 16, "bold"))
title_app.pack(pady=15)

model_label = tk.Label(root, text="Seleziona il modello YOLOv10:", font=("Courier", 11))
model_label.pack()
selected_model = tk.StringVar()
model_menu = ttk.Combobox(root, textvariable=selected_model, state="readonly", font=("Courier", 10))
model_menu['values'] = ('YOLOv10n', 'YOLOv10s', 'YOLOv10m')
model_menu.pack(pady=5)
model_menu.bind("<<ComboboxSelected>>", load_model)

button_frame = tk.Frame(root)
button_frame.pack(pady=15)

load_btn = tk.Button(button_frame, text="Carica Immagine", command=load_image, 
                    width=18, font=("Courier", 10), bg="lightblue")
load_btn.pack(side=tk.LEFT, padx=5)

predict_btn = tk.Button(button_frame, text="Esegui Predizione", command=predict, 
                       width=18, font=("Courier", 10), bg="lightgreen")
predict_btn.pack(side=tk.LEFT, padx=5)

remove_btn = tk.Button(button_frame, text="Rimuovi Immagine", command=remove_image, 
                      width=18, font=("Courier", 10), bg="lightcoral", state=tk.DISABLED)
remove_btn.pack(side=tk.LEFT, padx=5)

image_frame = tk.Frame(root)
image_frame.pack(pady=20)

title_left = tk.Label(image_frame, text="Immagine Originale", font=("Courier", 12, "bold"))
title_left.grid(row=0, column=0, padx=20, pady=10)
title_left.grid_remove()

title_center = tk.Label(image_frame, text="Ground Truth", font=("Courier", 12, "bold"))
title_center.grid(row=0, column=1, padx=20, pady=10)
title_center.grid_remove()

title_right = tk.Label(image_frame, text="Predizioni YOLO", font=("Courier", 12, "bold"))
title_right.grid(row=0, column=2, padx=20, pady=10)
title_right.grid_remove()

img_panel_left = tk.Label(image_frame, relief="sunken", bd=3, image=empty_img, 
                         width=400, height=300, bg="lightgray")
img_panel_left.grid(row=1, column=0, padx=20, pady=10)

img_panel_center = tk.Label(image_frame, relief="sunken", bd=3, image=empty_img, 
                           width=400, height=300, bg="lightgray")
img_panel_center.grid(row=1, column=1, padx=20, pady=10)

img_panel_right = tk.Label(image_frame, relief="sunken", bd=3, image=empty_img, 
                          width=400, height=300, bg="lightgray")
img_panel_right.grid(row=1, column=2, padx=20, pady=10)

root.mainloop()