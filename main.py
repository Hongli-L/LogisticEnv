import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Image Feature Extraction and Logistic Regression Training')
    parser.add_argument('--features', type=str, default=None,
                        help='Path to the features file (npy format). If not provided, features will be extracted.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the pre-trained model file. If not provided, a new model will be trained.')
    return parser.parse_args()

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # 增加 batch 维度
    return img_tensor

def extract_features(image_path, pre_model):
    img_tensor = load_image(image_path)
    with torch.no_grad():  # 不计算梯度
        features = pre_model(img_tensor)
    return features.squeeze().numpy()  # 返回特征向量并去掉多余的维度

def save_features(features_dict, save_path):
    np.save(save_path, features_dict)

def load_features(load_path):
    return np.load(load_path, allow_pickle=True).item()

def main():
    global preprocess  # 定义为全局变量
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pre_model = models.resnet50(pretrained=True)
    pre_model = nn.Sequential(*list(pre_model.children())[:-1])
    pre_model.eval()

    # 提取 UC Merced Land Use Dataset 的图像特征
    data_folder = 'data/UCMerced_LandUse/Images'  # 数据集图像所在文件夹
    features_dict = {}
    features_file = 'uc_merced_features_with_labels_test.npy'
    model_file = 'logistic_model_test.pkl'

    args = get_args()
    if args.features:
        features_file = args.features  # 使用用户指定的特征文件
    if args.model:
        model_file = args.model  # 使用用户指定的模型文件

    if features_file is not None and os.path.exists(features_file):
        features_dict = load_features(features_file)
        print("Loaded features from existing file.")
    else:
        print("Extracting features\n")
        for label in os.listdir(data_folder):
            label_folder = os.path.join(data_folder, label)
            if os.path.isdir(label_folder):
                for filename in os.listdir(label_folder):
                    if filename.endswith('.tif'):  # 只处理 .tif 图像
                        image_path = os.path.join(label_folder, filename)
                        features = extract_features(image_path, pre_model)
                        features_dict[filename] = {
                            'features': features,
                            'label': label
                        }
        save_features(features_dict, features_file)
        print("Extracted and saved features with labels for images in UC Merced Land Use Dataset.")

    # 准备特征和标签数据
    data = load_features(features_file)
    features = np.array([entry['features'] for entry in data.values()])
    labels = np.array([entry['label'] for entry in data.values()])

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_file is not None and os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            loaded_model = pickle.load(f)
        print("Loaded existing model.")

        y_pred = loaded_model.predict(X_test)

    else:
        # 训练逻辑回归模型
        print("Training new model\n")
        Logisticmodel = LogisticRegression(max_iter=1000, solver='saga')
        Logisticmodel.fit(X_train, y_train)
        print("Finished training")
        with open(model_file, 'wb') as f:
            pickle.dump(Logisticmodel, f)
        # 预测
        y_pred = Logisticmodel.predict(X_test)

    # 模型评估
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)

if __name__ == "__main__":
    main()
