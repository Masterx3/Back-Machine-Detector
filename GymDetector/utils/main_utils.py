import os
import sys
import yaml
import base64
from GymDetector.exception import AppException
from GymDetector.logger import logging
import requests
import zipfile
import cv2
import matplotlib.pyplot as plt
import glob
import random
from PIL import Image # for image manipulation
import shutil
import hashlib
from shutil import move
import json  # for annotation conversion

# Read YAML files
def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            logging.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise AppException(e, sys) from e
    
# Write YAML files
def write_yaml(file_path: str, content, replace: bool) -> dict:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        with open(file_path) as f:
            yaml.dump(content, f)
            f.close()
            
        logging.info(f"YAML written in {file_path}")
        
    except Exception as e:
        raise AppException(e, sys) from e

# Decode image from binary back into an image
def decode_img(imgstring, filename):
    img_data = base64.b64decode(imgstring)
    
    with open('./data' + filename, 'wb') as f:
        f.write(img_data)
        f.close()

# Encode image to binary
def encode_img(cropped_image_path):

    with open(cropped_image_path, 'rb') as f:
        return base64.b64encode(f.read())
        
# Download an online url's file:
def download_file(url, save_name):
    os.makedirs('data', exist_ok=True)
    url = url
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)
        print(f"Downloaded:{save_name}")
    else:
        print('File already exists.')
        
# Unzip the datasets files:
def unzip(zip_file=None):
    try:
        with zipfile.ZipFile(zip_file) as z:
            os.makedirs('data', exist_ok=True)
            z.extractall("./data")
            print("Extracted all in data folder")
    except:
        print("Invalid file format")

# Converts bounding boxes in YOLO format to xmin, ymin, xmax, ymax:
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax

def plot_box(image, bboxes, labels):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # Denormalize the coordinates.
        xmin = int(x1*w)
        ymin = int(y1*h)
        xmax = int(x2*w)
        ymax = int(y2*h)

        thickness = max(2, int(w/275))
                
        cv2.rectangle(
            image, 
            (xmin, ymin), (xmax, ymax),
            color=(0, 0, 255),
            thickness=thickness
        )
    return image

# Plots YOLO labeled images with their bounding boxes:
def plot_yolo(image_paths, label_paths, num_samples):
    all_images = []
    all_images.extend(glob.glob(image_paths+'/*.jpg'))
    all_images.extend(glob.glob(image_paths+'/*.JPG'))
    
    all_images.sort()

    num_images = len(all_images)
    
    # Calculate the number of rows and columns for subplots
    num_rows = (num_samples + 1) // 2
    num_cols = min(2, num_samples)

    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        j = random.randint(0, num_images - 1)
        image_name = all_images[j]
        image_name = '.'.join(image_name.split(os.path.sep)[-1].split('.')[:-1])
        image = cv2.imread(all_images[j])
        with open(os.path.join(label_paths, image_name+'.txt'), 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c, y_c, w, h = bbox_string.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.tight_layout()
    plt.show()


# Overlay 3d models and transparent images on realistic gym backgrounds: (Didn't prove useful)
def overlay_images(src_folder, bg_folder, output_folder):

  for src_name in os.listdir(src_folder):

    foreground = Image.open(os.path.join(src_folder, src_name)) 
    fg_width, fg_height = foreground.size

    for i in range(25):

      bg_name = random.choice(os.listdir(bg_folder))
      background = Image.open(os.path.join(bg_folder, bg_name))

      bg_width, bg_height = background.size

      if bg_width < fg_width or bg_height < fg_height:
        # Resize bg up to size of foreground
        background = background.resize((fg_width, fg_height))  

      else:
        # Crop bg to size of foreground
        left = (bg_width - fg_width) / 2
        top = (bg_height - fg_height) / 2
        right = (bg_width + fg_width) / 2
        bottom = (bg_height + fg_height) / 2
        background = background.crop((left, top, right, bottom))

      # Paste and save output image
      foreground = foreground.convert("RGBA")
      background.paste(foreground, (0, 0), foreground)

      name, ext = os.path.splitext(src_name)
      new_name = f"{name}_{i}{ext}"
      new_path = os.path.join(output_folder, new_name)
      background.save(new_path)

  print("Overlay complete!")

# duplicate annotations with names same to augmented images

def duplicate_text_files(text_folder, image_folder):

  for text_file in os.listdir(text_folder):

    name, _ = os.path.splitext(text_file)

    for image in os.listdir(image_folder):
      
      if name in image:
      
        # Remove .png extension 
        image_name = os.path.splitext(image)[0]  

        text_path = os.path.join(text_folder, image_name + '.txt')
        
        if not os.path.exists(text_path):
          
          shutil.copy(os.path.join(text_folder, text_file), text_path)

  print("Duplication complete!")

# Finds images that don't have a corresponding label
def find_and_move_missing_images(images_folder, text_folder, missing_folder):
    
    if not (os.path.isdir(text_folder) and os.path.isdir(images_folder)):
        return 'incorrect path'
    
    missing = []
    
    for filename in os.listdir(images_folder):
        name, ext = os.path.splitext(filename)


            
        text_path = os.path.join(text_folder, name + '.txt')
            
        if not os.path.exists(text_path):
            image_path = os.path.join(images_folder, filename)  
            missing.append(filename)
            move(image_path, missing_folder)
        
    return missing

# Finds labels that don't have a corresponding image
def find_and_move_missing_txts(images_folder, text_folder, missing_folder):
    if not (os.path.isdir(text_folder) and os.path.isdir(images_folder)):
        return 'incorrect path'
    
    missing = []
    
    for f in os.listdir(text_folder):
        if f.endswith('.txt'):
            text_path = os.path.join(text_folder, f)
            image_name = os.path.splitext(f)[0]
            image_extensions = ['.png', '.jpg', '.webp', '.jpeg', '.PNG', '.JPG', '.WEBP', '.JPEG']

            image_found = False
            for ext in image_extensions:
                image_path = os.path.join(images_folder, image_name + ext)
                if os.path.exists(image_path):
                    image_found = True
                    break

            if not image_found:
                missing.append(f)
                shutil.move(text_path, os.path.join(missing_folder, f))
    
    return missing

# extracts annotations and images for specific classes from yolo v8 format datasets
def extract_files(txt_folder, image_folder, number_list):
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        txt_path = os.path.join(txt_folder, txt_file)
        
        with open(txt_path, 'r') as f:
            line = f.readline().strip()
            line_values = line.split()
            
            if len(line_values) > 0:
                first_number = int(line_values[0])
                
                if first_number in number_list:
                    image_file = os.path.splitext(txt_file)[0] + '.jpg'
                    image_path = os.path.join(image_folder, image_file)
                    
                    output_folder = os.path.join(image_folder, str(first_number))
                    os.makedirs(output_folder, exist_ok=True)
                    
                    output_image_path = os.path.join(output_folder, image_file)
                    output_txt_path = os.path.join(output_folder, txt_file)
                    
                    shutil.copy(image_path, output_image_path)
                    shutil.copy(txt_path, output_txt_path)
    
    print("Extraction completed successfully.")
    
# Converts (AnyLabeling) .JSON labels to YOLO labels
def convert_to_yolo(input_folder, output_folder, class_mapping):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(input_folder, filename)
            with open(json_path, encoding='utf-8') as f:
                data = json.load(f)

            image_width = data['imageWidth']
            image_height = data['imageHeight']

            yolo_lines = []
            for shape in data['shapes']:
                label = shape['label']
                class_id = class_mapping[label]
                x_min, y_min = shape['points'][0]
                x_max, y_max = shape['points'][1]

                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                bbox_width = (x_max - x_min) / image_width
                bbox_height = (y_max - y_min) / image_height

                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                yolo_lines.append(yolo_line)

            output_txt = os.path.splitext(filename)[0] + '.txt'
            output_path = os.path.join(output_folder, output_txt)

            with open(output_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

# Changes the class of a YOLO labels folder
def change_first_num(path,new_num):
    
    if not os.path.isdir(path):
        return 'Path incorrect'
    
    files = os.listdir(path)
    txt_list = [f for f in files if f.endswith('.txt')]
    for txt in txt_list:
        f_path = os.path.join(path, txt)
        with open(f_path, 'r') as f:
            lines = f.readlines()
        with open(f_path, 'w') as f:
            for line in lines:
                values = line.strip().split()
                if len(values) > 0:
                    values[0] = str(new_num)
                    new_line = ' '.join(values) + '\n'
                    f.write(new_line)
                    
# Checks for all categories inside a YOLO labels folder
def unique_categories(folder_path, num):
    if not os.path.isdir(folder_path):
        return 'wrong path'
    unique = set()
    num_paths = []
    num_str = str(num)  # Convert the num to a string
    for txt in os.listdir(folder_path):
        if txt.endswith('.txt'):
            file_path = os.path.join(folder_path, txt)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    nums = line.strip().split()
                    first_num = str(nums[0])
                    unique.add(first_num)
                    if first_num.startswith(num_str):  # Use the string version of num
                        num_paths.append(txt)
    print(f"uniques: {unique}")
    print(f"with nums: {num_paths}")
    
# Plots images within a path to a 4xn grid using Matplotlib
def plot_4x(directory):
    files = os.listdir(directory)
    png_files = [file for file in files if file.lower().endswith('.png') or file.lower().endswith('.jpg')]

    num_images = len(png_files)
    cols = 4
    rows = (num_images + cols - 1) // cols

    dpi = 400  # Adjust the DPI value as per your needs

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), dpi=dpi)

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            image_path = os.path.join(directory, png_files[i])
            ax.imshow(plt.imread(image_path))
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
