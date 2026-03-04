#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# MOUNT GOOGLE DRIVE & IMPORT LIBRARIES
from google.colab import drive
drive.mount('/content/drive')
import os
import cv2
import numpy as np
from tqdm import tqdm
import zipfile

# DEFINE PATHS & IMAGE SETTINGS
INPUT_DIR  = "/content/drive/MyDrive/CC Pap Image final dataset"
OUTPUT_DIR = "/content/cervical_clear_jpg"
ZIP_PATH   = "/content/cervical_clear_dataset.zip"
IMG_SIZE = 224  # You can change to 256 or 512

# VERIFY CLASS FOLDERS (IMPORTANT)
CLASSES = [d for d in os.listdir(INPUT_DIR)
           if os.path.isdir(os.path.join(INPUT_DIR, d))]
print("Detected classes:", CLASSES)

# NATURAL IMAGE ENHANCEMENT FUNCTION
def enhance_image_natural(img):
    # 1. Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # 2. Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # 3. CLAHE for contrast (safe for medical images)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # 4. Edge-preserving denoising
    img = cv2.fastNlMeansDenoisingColored(
        img, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21
    )
    # 5. Unsharp masking (clarity boost)
    gaussian = cv2.GaussianBlur(img, (0,0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    return img

   # PROCESS & SAVE IMAGES (CLASS-WISE)
os.makedirs(OUTPUT_DIR, exist_ok=True)
total = 0
for cls in CLASSES:
    in_path = os.path.join(INPUT_DIR, cls)
    out_path = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(out_path, exist_ok=True)
    for img_name in tqdm(os.listdir(in_path), desc=f"Processing {cls}"):
        img_path = os.path.join(in_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        enhanced = enhance_image_natural(img)
        save_name = os.path.splitext(img_name)[0] + ".jpg"
        cv2.imwrite(
            os.path.join(out_path, save_name),
            enhanced,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        total += 1
print("✅ Total processed images:", total)
# CREATE ZIP FROM THE FOLDER
import shutil
shutil.make_archive(
    base_name="/content/cervical_clear_jpg",
    format="zip",
    root_dir="/content",
    base_dir="cervical_clear_jpg"
)
print("ZIP created at /content/cervical_clear_jpg.zip")

# DOWNLOAD THE ZIP FILE
from google.colab import files
files.download("/content/cervical_clear_jpg.zip")

