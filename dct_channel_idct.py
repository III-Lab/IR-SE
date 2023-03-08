

import numpy as np
import os
import cv2
import torch
from squeeze_and_excitation import ChannelSpatialSELayer

'''
input: numpy[256,256,3]
output: numpy[256,256,3]
'''
def DctChannelIDct(image):
    img = image[:,:,0] 
    img8_8 = np.zeros((8,8,1024), np.float32)       # 创建一个空图像
    ch = 0
    for i in range(0, 256, 8):                  # [256,256] -> [8,8,1024]
        for j in range(0, 256, 8):
            slice_img = img[i:i+8, j:j+8] # [行 列] 从左至右，从上至下
            img8_8[:,:,ch] = np.array(slice_img)
            ch = ch+1

    img8_8 = np.float32(img8_8) # 转成float类型才能进行dct变换
    dct_img = np.zeros_like(img8_8, np.float32) 
    for c in range(1024):       # 遍历每一个通道进行dct变换
        dct_img[:,:,c] = cv2.dct(img8_8[:,:,c])

    fre_img = np.zeros((32,32,64), np.float32)  
    ch = 0
    for i in range(8):
        for j in range(8):
            _temp = np.zeros((32,32),np.float32) 
            channel_data = dct_img[i, j, :] # 获取相同频率的点集合
            for k in range(32):
                _temp[:,k] = channel_data[k*32:(k+1)*32] # 从上至下排列像素
            fre_img[:, :, ch] = _temp   # 赋值给频率图
            ch = ch+1
    
    fre_img_trans = np.transpose(fre_img, [2, 0, 1]) # 转置
    fre_img_tensor = torch.from_numpy(fre_img_trans) # numpy -> tensor
    img_tensor = fre_img_tensor.unsqueeze(0) # add batch dimension torch.Size([1, 64, 32, 32])

    csse_layer = ChannelSpatialSELayer(num_channels=64, reduction_ratio=2) # 通道注意力
    csse_img = csse_layer(img_tensor) 

    csse_tensor = csse_img.squeeze()
    csse_numpy = csse_tensor.detach().numpy()   # tensor -> numpy
    csse_array = np.transpose(csse_numpy, [1, 2, 0]) # 转置 H W C

    img8_8 = np.zeros((8,8,1024), np.float32)
    for ch in range(64):                        # [32,32,64] -> [8,8,1024]
        _temp = np.zeros((1024,), np.float32)
        ch_data = csse_array[:,:,ch]
        for i in range(32):
            _temp[i*32:(i+1)*32] = ch_data[:,i]
        row = ch//8
        col = ch%8
        img8_8[row, col, :] = _temp
    
    idct_img = np.zeros((8,8,1024), np.float32)     # idct 
    for c in range(1024):
        idct_img[:,:,c] = cv2.idct(img8_8[:,:,c])
    
    out_img = np.zeros((256,256), np.float32)       # [8,8,1024] -> [256,256]
    ch = 0
    for i in range(0,256,8):
        for j in range(0,256,8):
            out_img[i:i+8,j:j+8] =  idct_img[:,:,ch]
            ch = ch+1
    
    out = np.zeros((256,256,3), np.uint8)
    for i in range(3):
        out[:,:,i] = out_img

    return out


if __name__ == '__main__':
    images_path = './dataset/images'
    labels_path = './dataset/labels'

    img_path = os.path.join(images_path, '1.bmp')
    img = cv2.imread(img_path)
    out = DctChannelIDct(img)
    cv2.imwrite("./out.bmp",out)
    cv2.imshow("out", out)
    cv2.waitKey(0)


    
