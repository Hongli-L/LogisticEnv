import torch
import os
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

model = models.resnet50(pretrained=True)

# 去掉全连接层
model = nn.Sequential(*list(model.children())[:-1])

# 模型设为评估模式
model.eval()

# 图像预处理
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 加载数据
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


# 提取特征
def extract_features(image_path):
    img_tensor = load_image(image_path)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy()

# 保存特征
def save_features(features_dict, save_path):
    np.save(save_path, features_dict)

# 加载特征
def load_features(load_path):
    return np.load(load_path, allow_pickle=True).item()

data_folder = 'data/UCMerced_LandUse/Images'
features_dict = {}

features_file = 'uc_merced_features.npy'

if os.path.exists(features_file):
    features_dict = load_features(features_file)
    print("features is exist")
else:
    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                if filename.endswith('.tif'):
                    image_path = os.path.join(label_folder, filename)
                    features = extract_features(image_path)
                    features_dict[filename] = features
    save_features(features_dict, features_file)
    print("finish extract")



