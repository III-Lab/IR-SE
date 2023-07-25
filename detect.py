import cv2
import torch
import torchvision.transforms as transforms
import os
import numpy as np
import sklearn.preprocessing as sp
from model import KeyPointModel
import matplotlib.pyplot as plt
from time import *

class KeyPointDetector:
    def __init__(self, weight_path, device='cpu'):
        # 初始化模型和权重路径
        self.device = device
        self.net = None
        self.weight_path = weight_path

        # 初始化数据预处理操作
        self.transforms_test = None

        # 加载模型和权重
        self.load_model()

    def load_model(self):
        # 构造预处理操作
        self.transforms_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4372, 0.4372, 0.4373],
                                 std=[0.2479, 0.2475, 0.2485])
        ])
        
        # 加载模型权重
        self.net = KeyPointModel().to(self.device)
        if os.path.exists(self.weight_path):
            try:
                self.net.load_state_dict(torch.load(self.weight_path, map_location=self.device))
            except:
                print("Error loading model weights from file: {}".format(self.weight_path))
        else:
            print("Model weights file not found: {}".format(self.weight_path))

    def detect(self, image_path):
        img_tensor_list = []
        origin = cv2.imread(image_path)
        origin1 = cv2.imread(image_path)

        # 图像预处理
        img = self.transforms_test(origin)
        img_tensor_list.append(img)
        img1 = self.transforms_test(origin1)
        img_tensor_list.append(img1)
        img_tensor_list = torch.stack(img_tensor_list, 0)

        self.net.eval()

        # 模型预测
        with torch.no_grad():
            pred = self.net(img_tensor_list[0:1].cpu())

        heatmap = pred.squeeze().cpu()
        hm = heatmap.detach().numpy()

        # 将数组按照一定规则归一化
        hm_normalized = self.normalization(hm)

        # 根据阈值进行二值化
        binarizer = sp.Binarizer(threshold=0.5)
        hm_binary = binarizer.transform(hm_normalized)

        # 转换为 uint8 格式的灰度图像
        hm_gray = np.uint8(255 * hm_binary)
        hm_gray_resized = cv2.resize(hm_gray, (256,256))
        origin_resized = cv2.resize(origin, (256,256))

        return hm_gray_resized, origin_resized

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range


if __name__ == '__main__':
    
    weight_path = './weight/epoch_98_0.081.pt'
    img = '1.bmp'

    test_image_path = './dataset/images'
    out_image_path = './output'
    img_path = os.path.join(test_image_path, img)
    kdetect = KeyPointDetector(weight_path)

    begin_time = time()
    hm, orgin = kdetect.detect(img_path)
    end_time = time()
    elapsed_time = end_time - begin_time
    print("Total time elapsed: %.2f seconds" % elapsed_time)
    hm = hm *2
    # cv2.imshow('out', hm)
    # cv2.imshow('orgin', orgin)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    out_name = os.path.join(out_image_path, img)
    cv2.imwrite(out_name, hm)