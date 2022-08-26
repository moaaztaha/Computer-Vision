import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def str2img(row):
    """Extracting images from dataframe row"""
    img_arr = np.fromstring(row.Image, dtype='int32', sep=' ').astype(np.int32)
    img = Image.fromarray(img_arr.reshape(-1, 96)).convert('L')
    return img


def row2points(row):
    """Extract and format points from dataframe row"""
    points = row[:-1].to_numpy().reshape(-1, 2).astype("float")
    return points


def prep_landmarks(df, index=1):
    img_name = df.iloc[index, 0]
    landmarks = df.iloc[index, 1:]
    landmarks = np.array(landmarks).astype("float")
    landmarks = landmarks.reshape(-1, 2)
    print("Image name: {}".format(img_name))
    print("Landmarks shape: {}".format(landmarks.shape))
    print("First 4 Landmarks: {}".format(landmarks[:4]))
    return img_name, landmarks


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image, cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')


def load_img(path, infer=False):
    """Load image and prepare it for inference"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
    img = cv2.resize(img, (224, 224))

    if infer:
        img = torch.tensor(img).permute(2, 0, 1)
        img = normalization(img).float()
        return img.to(device)
    return img
