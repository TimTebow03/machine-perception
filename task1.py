

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

import os
from ultralytics import YOLO
import cv2 as cv2
import glob

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


def run_task1(image_path, config):
    model = YOLO("train8Small.pt")
    image_files = glob.glob(os.path.join(image_path, '*.jpg'))

    for imgNum, img_file in enumerate(image_files, start=1):
        #Resized to match training image size
        results = model(img_file, imgsz=640)
        img = cv2.imread(img_file)

        CONF_THRESH = 0.3
        best_box = None
        best_conf = 0.0

        # Finding Highest confidence box
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf)
                if conf >= CONF_THRESH and conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].tolist()

        # Save crop of the highest confidence box if found
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box)
            cropped_num = img[y1:y2, x1:x2]
            output_path = f"output/task1/bn{imgNum}.png"
            save_output(output_path, cropped_num, output_type='image')
            print(f"Saved highest confidence crop for img{imgNum} with conf {best_conf:.2f}")
        else:
            print(f"No detection above threshold for img{imgNum}")