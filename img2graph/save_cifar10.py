import os
import random
import cv2
import h5py
from multiprocessing import Process, Manager
from torchvision import datasets
from img2graph import granular_balls_generate
import numpy as np  # 确保导入 numpy

# CIFAR-10 类别映射
CLASSES = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}

def save_image2h5py_multiprocessing(input_data, output_dir):
    for img, label, image_name in input_data:
        x, adj, edge_attr, center_index = ball2graph_rgb_hsv(img)

        # Create output path
        set_class_dir = os.path.join(output_dir, str(label))
        createFile(set_class_dir)

        # Save data to h5py file
        with h5py.File(os.path.join(set_class_dir, f"{image_name}.h5"), 'w') as f:
            f['x'] = x
            f['adj'] = adj
            f['y'] = label
            f['edge_attr'] = edge_attr

def createFile(filePath):
    if not os.path.exists(filePath):
        os.makedirs(filePath)
        print(f'Created directory: {filePath}')

def process_data(dataset, output_dir, set_name):
    data = [
        (np.array(dataset[i][0]), dataset[i][1], f"{set_name}_{i}") for i in range(len(dataset))
    ]

    random.shuffle(data)
    coreNum = 50
    lenPerSt = int(len(data) / coreNum + 1)
    paths = [data[i * lenPerSt:(i + 1) * lenPerSt] for i in range(coreNum)]

    jobs = []
    for i in range(coreNum):
        p = Process(target=save_image2h5py_multiprocessing, args=(paths[i], os.path.join(output_dir, set_name)))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

if __name__ == '__main__':

    data_dir = './cifar10_data'
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)

    output_dir = './cifar10_h5'
    os.makedirs(output_dir, exist_ok=True)

    # 处理训练集和验证集
    process_data(train_dataset, output_dir, 'train')
    process_data(val_dataset, output_dir, 'val')
