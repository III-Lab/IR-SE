
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
from squeeze_and_excitation import ChannelSpatialSELayer

class IRSE(nn.Module):
    def __init__(self, bs, num_channels, reduction_ratio) -> None:
        super(IRSE, self).__init__()
        self.se_layer = ChannelSpatialSELayer(num_channels, reduction_ratio)
        self.bs = bs
    
    '''
    numpy to pytorch
    pytorch to numpy
    '''
    def se_transpose(self, bs, x): 
        if type(x) == np.ndarray:       # [H,W,C] -> [B,C,H,W]
            x = np.transpose(x, [2,0,1]) 
            x_out = torch.from_numpy(x)
            x_out.unsqueeze(0)
            x_out = x_out.repeat(bs,1,1,1) 
        elif type(x) == torch.Tensor:   # [B,C,H,W] -> [H,W,C]
            x = x[0]
            x_out = x.detach().numpy()
            x_out = np.transpose(x_out, [1, 2, 0])
        return x_out
    
    '''
    input: <class 'numpy.ndarray'> (256, 256, 3)
    output: <class 'numpy.ndarray'> (8, 8, 1024)
    '''
    def se_dct(self, x):
        img = x[:,:,0] 
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
        return dct_img
    
    '''
    input: <class 'numpy.ndarray'> (8, 8, 1024)
    output: <class 'numpy.ndarray'> (32, 32, 64)
    '''
    def se_fre(self, x):
        fre = np.zeros((32,32,64), np.float32)
        ch = 0
        for i in range(8):
            for j in range(8):
                _temp = np.zeros((32,32),np.float32) 
                channel_data = x[i, j, :] # 获取相同频率的点集合
                for k in range(32):
                    _temp[:,k] = channel_data[k*32:(k+1)*32] # 从上至下排列像素
                fre[:, :, ch] = _temp   # 赋值给频率图
                ch = ch+1
        return fre
    
    '''
    input: <class 'numpy.ndarray'> (32, 32, 64)
    output: <class 'numpy.ndarray'> (8, 8, 1024)
    '''
    def se_ifre(self, x):
        array = np.zeros((8,8,1024), np.float32)
        for ch in range(64):                       
            _temp = np.zeros((1024,), np.float32)
            ch_data = x[:,:,ch]
            for i in range(32):
                _temp[i*32:(i+1)*32] = ch_data[:,i]
            row = ch//8
            col = ch%8
            array[row, col, :] = _temp
        return array

    '''
    dct 反变换
    '''
    def se_idct(self, x):
        idct_array = np.zeros((8,8,1024), np.float32)     # idct 
        for c in range(1024):
            idct_array[:,:,c] = cv2.idct(x[:,:,c])
        
        out_array = np.zeros((256,256), np.float32)       # [8,8,1024] -> [256,256]
        ch = 0
        for i in range(0,256,8):
            for j in range(0,256,8):
                out_array[i:i+8,j:j+8] =  idct_array[:,:,ch]
                ch = ch+1
        return out_array
    
    '''
    图像 2d 转 3d
    '''
    def se_2dto3d(self, x):
        return np.repeat(x[:,:,np.newaxis], 3, axis=2)

    '''
    前向传播函数
    '''
    def forward(self, x):                   # <class 'torch.Tensor'> torch.Size([2, 3, 256, 256])
        x = self.se_transpose(self.bs, x)   # <class 'numpy.ndarray'> (256, 256, 3)
        x = self.se_dct(x)                  # <class 'numpy.ndarray'> (8, 8, 1024)
        x = self.se_fre(x)                  #<class 'numpy.ndarray'> (32, 32, 64)
        x = self.se_transpose(self.bs, x)   # <class 'torch.Tensor'> torch.Size([2, 64, 32, 32])
        x = self.se_layer(x)                # <class 'torch.Tensor'> torch.Size([2, 64, 32, 32])
        x = self.se_transpose(self.bs, x)   # <class 'numpy.ndarray'> (32, 32, 64)
        x = self.se_ifre(x)                 # <class 'numpy.ndarray'> (8, 8, 1024)
        x = self.se_idct(x)                 # <class 'numpy.ndarray'> (256, 256)
        x = self.se_2dto3d(x)               # <class 'numpy.ndarray'> (256, 256, 3)
        # cv2.imwrite("./out1.bmp",x)
        x = self.se_transpose(self.bs, x)   # <class 'torch.Tensor'> torch.Size([2, 3, 256, 256])

        return x

if __name__ == '__main__':
    images_path = './dataset/images'
    labels_path = './dataset/labels'

    img_path = os.path.join(images_path, '1.bmp')
    img = cv2.imread(img_path)
    app = IRSE(2,64,2)
    tensor = app.se_transpose(2, img)
    out = app(tensor)
    print(type(out), out.shape)
