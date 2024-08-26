import os
import sys
import io
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

from GymDetector.utils import yolov3_config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    
    with open(os.path.join('./data', filename), 'wb') as f:
        f.write(img_data)
        f.close()

def decode_img2(image_data, filename):
  # Convert bytes to a PIL image object
  image = Image.open(io.BytesIO(image_data))

  # Save the image to a temporary file
  image.save(filename)
    
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

    dpi = 400

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

def yolo_to_coco(yolo_dir, output_json_train, output_json_val, config_path):
    def process_phase(phase):
        images = []
        annotations = []
        img_id = 0
        annotation_id = 0

        img_list = os.listdir(os.path.join(yolo_dir, "images", phase))
        for img_name in img_list:
            img_path = os.path.join(yolo_dir, "images", phase, img_name)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except IOError:
                print(f"Error opening image {img_path}. Skipping.")
                continue

            images.append({
                "id": img_id,
                "width": width,
                "height": height,
                "file_name": os.path.join("images", phase, img_name)
            })

            label_path = os.path.join(yolo_dir, "labels", phase, os.path.splitext(img_name)[0] + ".txt")

            try:
                with open(label_path, "r") as file:
                    lines = file.readlines()
            except IOError:
                print(f"Error reading label file {label_path}. Skipping.")
                continue

            for line in lines:
                try:
                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
                except ValueError:
                    print(f"Error parsing line in {label_path}. Skipping.")
                    continue

                x_center *= width
                y_center *= height
                bbox_width *= width
                bbox_height *= height
                x_min = x_center - bbox_width / 2
                y_min = y_center - bbox_height / 2

                annotations.append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": int(class_id),
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                annotation_id += 1

            img_id += 1

        return images, annotations

    # Read class names from config file
    with open(config_path, 'r') as f:
        class_data = yaml.safe_load(f)
    class_names = [class_data['names'][key] for key in sorted(class_data['names'].keys())]

    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]

    # Process train and val sets separately
    train_images, train_annotations = process_phase("train")
    val_images, val_annotations = process_phase("val")

    # Save train annotations
    with open(output_json_train, 'w') as outfile:
        json.dump({
            "images": train_images,
            "annotations": train_annotations,
            "categories": categories
        }, outfile, indent=4)

    # Save val annotations
    with open(output_json_val, 'w') as outfile:
        json.dump({
            "images": val_images,
            "annotations": val_annotations,
            "categories": categories
        }, outfile, indent=4)
        
def clean_img_dir_and_labels(image_dir):
    corrupted = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            image_path = os.path.join(root, file)
            try:
                with Image.open(image_path) as img:
                    img.load()
            except(IOError, OSError) as e:
                corrupted.append(image_path)
                os.remove(image_path)
                ann_path = image_path.replace('images', 'labels')
                os.remove(ann_path)
    return "The following corrupted images have been removed:" + "\n".join(corrupted)

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """
    
    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=7
):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI

    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def plot_image(image, boxes, config_path):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = list(read_yaml(config_path)['names'].values())
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    if len(boxes)>0:
        for box in boxes:
            
            assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
            class_pred = box[0]
            box = box[2:]
            upper_left_x = box[0] - box[2] / 2
            upper_left_y = box[1] - box[3] / 2
            rect = patches.Rectangle(
                (upper_left_x * width, upper_left_y * height),
                box[2] * width,
                box[3] * height,
                linewidth=2,
                edgecolor=colors[int(class_pred)],
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)
            plt.text(
                upper_left_x * width,
                upper_left_y * height,
                s=class_labels[int(class_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": colors[int(class_pred)], "pad": 0},
            )

    plt.show()

def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()

def check_class_accuracy(model, loader, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for x, y in tqdm(loader):
        x = x.to(yolov3_config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(yolov3_config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()

def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=yolov3_config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)



def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False