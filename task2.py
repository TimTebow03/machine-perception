

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: Matthew Henderson-Kelly
# Last Modified: 2024-09-09

import numpy as np
import cv2 as cv2
import os
import glob
import math
from imutils.object_detection import non_max_suppression

def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")

def resize_keep_aspect(image, target_area=50000):
    #as we round to int, the result is ever so slightly off 50000
    h, w = image.shape[:2]
    current_area = h * w
    ratio = w/h
    new_h = int((target_area/ratio) ** 0.5)
    new_w = int(target_area/new_h)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

# Stretching the image horizontally helps separate
# the digits by widening the gap between them, also generally numbers
# are captured above the camera, this perspective naturally warps the digits
# such that they grow vertically, this crudely counteracts this a little.
def stretch_image(image,strW=1, strH=1):
    h, w = image.shape[:2]
    str_w = int(w * strW)
    str_h = int(h * strH)
    stretched = cv2.resize(image, (str_w, str_h), interpolation=cv2.INTER_LINEAR)
    return stretched


def draw_mser_filtered(img, base_name):
    imgHeight, imgWidth = img.shape[:2]
    mser = cv2.MSER_create()
    mser.setDelta(5) #trial and error, 5 worked best
    mser.setMinArea(int(0.05 * imgHeight * imgWidth)) 
    mser.setMaxArea(int(0.32 * imgHeight * imgWidth))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    regions, _ = mser.detectRegions(gray)

    boxes = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        boxes.append([x, y, x + w, y + h])

    filtered_boxes = []
    for (x1, y1, x2, y2) in boxes:
        w = x2 - x1
        h = y2 - y1
        if 0.3 < w/h < 4 and w < imgWidth/2.5:
            filtered_boxes.append([x1, y1, x2, y2])

    filtered_boxes = np.array(filtered_boxes)
    final_boxes = non_max_suppression(filtered_boxes, probs=None, overlapThresh=0.2)

    for i, (x1, y1, x2, y2) in enumerate(final_boxes):
        crop = img[y1:y2, x1:x2] 
        output_path = f"output/task2/{base_name}/c{i+1}.png"
        save_output(output_path, crop, "image")

def run_task2(image_path, config):
    # TODO: Implement task 2 here
    # ------------------------------------- #
    print(image_path)
    print(os.listdir(image_path))
    processed_images = []
    image_files = glob.glob(os.path.join(image_path, '*.png'))
    for file_name in image_files:
        filename = os.path.basename(file_name)
        base_name, ext = os.path.splitext(filename)
        img = cv2.imread(file_name)
        print("check 1")
        resized = resize_keep_aspect(img)
        print("check 2")
        stretched = stretch_image(resized, 1.6, 1)
        print("check 3")
        smoothed = cv2.bilateralFilter(stretched, d=1, sigmaColor=70, sigmaSpace=70)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        closed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
        processed_images.append(closed)
        draw_mser_filtered(closed, base_name)
    # ------------------------------------- #
    output_path = f"output/task2/result.txt"
    save_output(output_path, "Task 2 output", output_type='txt')
