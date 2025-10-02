

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


# Author: [Your Name]
# Last Modified: 2024-09-09

import os
import glob
from ultralytics import YOLO
import cv2 as cv2
from imutils.object_detection import non_max_suppression
import numpy as np
import math
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim

img_size = 32       # resize images to 32x32
batch_size = 32
num_classes = 10
epochs = 10
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*8*8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



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

def task1_detection(imgNum, img_file):
    model = YOLO("train8Small.pt")
    results = model(img_file, imgsz=640)
    img = cv2.imread(img_file)
    CONF_THRESH = 0.3
    best_box = None
    best_conf = 0.0

    # Loop through detections to find the highest confidence box
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf)
            if conf >= CONF_THRESH and conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

    # Save crop of the highest confidence box if found
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        cropped_num = img[y1:y2, x1:x2]
        output_path = f"output/task4/image{imgNum}/bn{imgNum}.png"
        save_output(output_path, cropped_num, output_type='image')
        print(f"Saved highest confidence crop for image {imgNum} with conf {best_conf:.2f}")
        return cropped_num
    else:
        print(f"No detection above threshold for image {imgNum}")

def stretch_image(image,strW=1, strH=1):
    h, w = image.shape[:2]
    str_w = int(w * strW)
    str_h = int(h * strH)
    stretched = cv2.resize(image, (str_w, str_h), interpolation=cv2.INTER_LINEAR)
    return stretched

def resize_keep_aspect(image, target_area=50000):
    h, w = image.shape[:2]
    current_area = h * w
    ratio = w/h
    new_h = int((target_area/ratio) ** 0.5)
    new_w = int(target_area/new_h)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def task2_localisation(imgNum, img):
    imgHeight, imgWidth = img.shape[:2]
    mser = cv2.MSER_create()
    mser.setDelta(5)
    mser.setMinArea(int(0.05 * imgHeight * imgWidth))
    mser.setMaxArea(int(0.32 * imgHeight * imgWidth))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    regions, _ = mser.detectRegions(gray)

    vis = img.copy()
    boxes = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        boxes.append([x, y, x + w, y + h])

    filtered_boxes = []
    for (x1, y1, x2, y2) in boxes:
        w = x2 - x1
        h = y2 - y1
        if 0.3 < w/h < 4 and w < imgWidth/2.5:  # width/height ratio reasonable
            filtered_boxes.append([x1, y1, x2, y2])

    #final_boxes = non_max_suppression(filtered_boxes, overlapThresh=0.2)
    filtered_boxes = np.array(filtered_boxes)
    final_boxes = non_max_suppression(filtered_boxes, probs=None, overlapThresh=0.2)

    final_cropped = []
    for i, (x1, y1, x2, y2) in enumerate(final_boxes):
        crop = img[y1:y2, x1:x2]
        final_cropped.append(crop)
        output_path = f"output/task4/image{imgNum}/c{i+1}.png"
        save_output(output_path, crop, "image")
    return final_cropped

def task3_classification(imgNum, i, img):
    model = TinyCNN()
    model.load_state_dict(torch.load("tinycnn.pth", map_location=device))
    model.to(device)
    model.eval()
    # print("Image Length:", len(img.shape))
    # channels = img.shape[2]
    # if channels == 3:
    #     print("3")
    # if channels == 4:
    #     print("4")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    # --- 2. Preprocessing transform ---
    transform = transforms.Compose([
        transforms.Grayscale(),          # make sure it's 1-channel
        transforms.Resize((32, 32)),    # match training size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # for subdir in os.listdir(image_path):
    #     subdir_path = os.path.join(image_path, subdir)
    #     if not os.path.isdir(subdir_path):
    #         continue  # skip if it's not a directory

    #     # Process all .png files inside subdir
    #     for img_file in glob.glob(os.path.join(subdir_path, "*.png")):
    #         img = Image.open(img_file).convert("L")  # grayscale
    #         img = transform(img).unsqueeze(0).to(device)  # [1,1,32,32]

            # --- 4. Forward pass and prediction ---
    img = Image.fromarray(img)
    img = img.convert("L")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        predicted_digit = torch.argmax(output, dim=1).item()

    # --- 5. Save prediction to txt file ---
    #rel_path = os.path.relpath(img_file, image_path)   # e.g. "bn1/c1.png"
    #rel_txt = os.path.splitext(rel_path)[0] + ".txt"   # "bn1/c1.txt"
    output_path = f"output/task4/image{imgNum}/c{i+1}"
    #output_path = os.path.join(f"output/task4/image{imgNum}", rel_txt)

    # use save_output from your code
    save_output(output_path, str(predicted_digit), output_type="txt")

def run_task4(image_path, config):
    # TODO: Implement task 4 here
    # ------------------------------------- #
    
    model = YOLO("train8Small.pt")
    image_files = glob.glob(os.path.join(image_path, '*.jpg'))

    model = TinyCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for imgNum, img_file in enumerate(image_files, start=1):
        print(img_file)
        filename = os.path.basename(img_file)
        base_name, ext = os.path.splitext(filename)
        img = cv2.imread(img_file)
        print("Beginning Task 1: Detection")
        detected_num = task1_detection(imgNum, img_file)
        if detected_num is None:
            print("Nothing detected")
            continue

        print("Beginning Task 2: Localisation")
        final_cropped = task2_localisation(imgNum, detected_num)

        for i, digit in enumerate(final_cropped):
            task3_classification(imgNum, i, digit)


    # ------------------------------------- #
    # output_path = f"output/task4/result.txt"
    # save_output(output_path, "Task 4 output", output_type='txt')