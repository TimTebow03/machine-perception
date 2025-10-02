

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
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

img_size = 32
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

# Initialize model
model = TinyCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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


def run_task3(image_path, config):
    # TODO: Implement task 3 here
    # ------------------------------------- #
    model = TinyCNN()
    model.load_state_dict(torch.load("tinycnn.pth", map_location=device))
    model.to(device)
    model.eval()


    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    for subdir in os.listdir(image_path):
        subdir_path = os.path.join(image_path, subdir)
        if not os.path.isdir(subdir_path):
            continue 

        for img_file in glob.glob(os.path.join(subdir_path, "*.png")):
            img = Image.open(img_file).convert("L")
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img)
                predicted_digit = torch.argmax(output, dim=1).item()

            rel_path = os.path.relpath(img_file, image_path)
            rel_txt = os.path.splitext(rel_path)[0] + ".txt"
            output_path = os.path.join("output/task3", rel_txt)

            save_output(output_path, str(predicted_digit), output_type="txt")
