
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
from squeeze_and_excitation import ChannelSpatialSELayer,ChannelSELayer, SpatialSELayer,SELayer


class IRSE(nn.Module):
    '''
    param: se_block_type{ NONE, CSE, SSE, CSSE }
    '''
    def __init__(self, num_channels=64, reduction_ratio=2, se_block_type=SELayer.NONE) -> None:
        super(IRSE, self).__init__()

        self.se_block_type = se_block_type
        if self.se_block_type == SELayer.CSE.value:                                 # 通道挤压
            self.se_layer = ChannelSELayer(num_channels,reduction_ratio)
        elif self.se_block_type == SELayer.SSE.value:                               # 控件挤压
            self.se_layer = SpatialSELayer(num_channels,reduction_ratio)
        elif self.se_block_type == SELayer.CSSE.value:                              # 通道与空间挤压
            self.se_layer = ChannelSpatialSELayer(num_channels,reduction_ratio)

    '''
    numpy 与 pytorch 向量互转
    numpy to pytorch
    pytorch to numpy
    '''
    def se_transpose(self, x): 
        if type(x) == np.ndarray:       # [H,W,C] -> [1,C,H,W]
            x = np.transpose(x, [2,0,1]) 
            x_out = torch.from_numpy(x)
            x_out.unsqueeze(0)
            x_out = x_out.repeat(1,1,1,1) 
        elif type(x) == torch.Tensor:   # [1,C,H,W] -> [H,W,C]
            x = x.squeeze()
            x_out = x.detach().numpy()
            x_out = np.transpose(x_out, [1, 2, 0])
        return x_out
    
    '''
    dct 变换
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
    提取相同频率并放在一个通道上
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
    函数 se_fre 的逆运算
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
    图像 2d 转 3d, 复制3份
    '''
    def se_2dto3d(self, x):
        return np.repeat(x[:,:,np.newaxis], 3, axis=2)

    '''
    前向传播函数
    '''
    def forward(self, x):                                   # <class 'torch.Tensor'> torch.Size([bs, 3, 256, 256])
        bs,c,h,w = x.shape
        assert (h == 256 and w == 256 and c == 3)
        self.bs = bs                                        # get input tensor batch size
        out_list = []
        for index in range(self.bs):                        # iterate over each batch
            x1 = x[index]                                   # <class 'torch.Tensor'> torch.Size([3, 256, 256])
            x1 = self.se_transpose(x1)                      # <class 'numpy.ndarray'> (256, 256, 3)
            x1 = self.se_dct(x1)                            # <class 'numpy.ndarray'> (8, 8, 1024)
            x1 = self.se_fre(x1)                            # <class 'numpy.ndarray'> (32, 32, 64)
            x1 = self.se_transpose(x1)                      # <class 'torch.Tensor'> torch.Size([1, 64, 32, 32])
            if self.se_block_type != SELayer.NONE.value:    
                x1 = self.se_layer(x1)                      # <class 'torch.Tensor'> torch.Size([1, 64, 32, 32])
            x1 = self.se_transpose(x1)                      # <class 'numpy.ndarray'> (32, 32, 64)
            x1 = self.se_ifre(x1)                           # <class 'numpy.ndarray'> (8, 8, 1024)
            x1 = self.se_idct(x1)                           # <class 'numpy.ndarray'> (256, 256)
            x1 = self.se_2dto3d(x1)                         # <class 'numpy.ndarray'> (256, 256, 3)
            x1 = self.se_transpose(x1)                      # <class 'torch.Tensor'> torch.Size([1, 3, 256, 256])
            out_list.append(x1)
        out = torch.cat(out_list, dim=0)                    # <class 'torch.Tensor'> torch.Size([bs, 3, 256, 256])
        return out

if __name__ == '__main__':
    # tensor = torch.randn(2, 3, 256, 256)
    img = cv2.imread('./dataset/images/2.bmp')
    app = IRSE(64,2, 'CSSE')
    tensor = app.se_transpose(img)
    out = app(tensor)
    print(type(out), out.shape)

''' save dct image
out_array = np.zeros((256,256), np.float32)       # [8,8,1024] -> [256,256]
ch = 0
for i in range(0,256,8):
    for j in range(0,256,8):
        out_array[i:i+8,j:j+8] =  x1[:,:,ch]
        ch = ch+1
out_array = self.se_2dto3d(out_array)                         # <class 'numpy.ndarray'> (256, 256, 3)
cv2.imwrite('./output/DCT.bmp', out_array)
'''
