import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from thop import clever_format
from thop import profile
class SematicEmbbedBlock(nn.Module):
    def __init__(self, high_in_plane, low_in_plane, out_plane):
        super(SematicEmbbedBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(high_in_plane, out_plane, 3, 1, 1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3x31 = nn.Conv2d(low_in_plane, out_plane, 3, 1, 1)
        self.conv1x1 = nn.Conv2d(low_in_plane, out_plane, 1)
        self.conv1x11 = nn.Conv2d(low_in_plane, out_plane, 1)

    def forward(self, high_x, low_x):
        high_x = self.upsample(self.conv3x3(high_x))
        high_x= self.conv1x11(high_x)
        low_x=self.conv3x31(low_x)
        low_x = self.conv1x1(low_x)

        return high_x +low_x


class KeyPointModel(nn.Module):
    """
    downsample ratio=2
    """

    def __init__(self):
        super(KeyPointModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((2, 2))
        #self.droup=torch.nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(6, 12, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(12)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(12, 20, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(20)
        self.relu3 = nn.ReLU(True)
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(20, 40, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(40)
        self.relu4 = nn.ReLU(True)

        self.maxpool4 = nn.MaxPool2d((2, 2))

        self.conv5 = nn.Conv2d(40, 64, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(True)

        self.maxpool5 = nn.MaxPool2d((2, 2))

        self.conv6 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU(True)

        self.maxpool6 = nn.MaxPool2d((2, 2))

        self.conv7 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU(True)

        # self.conv8 = nn.Conv2d(64, 96, 3, 1, 1)
        # self.bn8 = nn.BatchNorm2d(96)
        # self.relu8 = nn.ReLU(True)
        # self.maxpool8 = nn.MaxPool2d((2, 2))

        self.resnet=nn.Sequential(
            Residual(6, 6),
            Residual(6, 6),
            Residual(6, 6),
        )
        self.resnet1 = nn.Sequential(
            Residual(12, 12),
            Residual(12, 12),
            Residual(12, 12),
        )
        # self.resnet2 = nn.Sequential(
        #     Residual(20, 20),
        #     Residual(20, 20),
        #     Residual(20, 20),
        #
        # )

        self.aspp=ASPP(256,[1,2,3])
        #self.non=NonLocalBlock(256)

        self.seb1 = SematicEmbbedBlock(256, 128, 128)
        self.seb2 = SematicEmbbedBlock(128, 64, 64)
        #self.seb3 = SematicEmbbedBlock(96, 64, 64)
        self.seb3 = SematicEmbbedBlock(64, 40, 40)
        self.seb4 = SematicEmbbedBlock(40, 20, 20)
        self.seb5 = SematicEmbbedBlock(20, 12, 12)
        self.seb6 = SematicEmbbedBlock(12, 6, 6)


        self.heatmap = nn.Conv2d(6, 1, 1)

    def forward(self, x):

        # print(x.shape) # torch.Size([2, 3, 256, 256])

        x1 = self.conv1(x)
        x1 = self.resnet(x1)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        m1 = self.maxpool1(x1)

        x2 = self.conv2(m1)
        x2 = self.resnet1(x2)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        m2 = self.maxpool2(x2)

        x3 = self.conv3(m2)
        #x3=self.resnet2(x3)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)

        m3 = self.maxpool3(x3)

        x4 = self.conv4(m3)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)

        m4 = self.maxpool4(x4)

        x5 = self.conv5(m4)
        x5 = self.bn5(x5)
        x5 = self.relu5(x5)

        # m8 = self.maxpool8(x5)
        #
        # x8 = self.conv8(m8)
        # x8 = self.bn8(x8)
        # x8 = self.relu8(x8)

        m5 = self.maxpool5(x5)

        x6 = self.conv6(m5)
        x6 = self.bn6(x6)
        x6 = self.relu6(x6)

        m6 = self.maxpool6(x6)
       # x6=self.aspp(x6)
        x7 = self.conv7(m6)
        x7 = self.bn7(x7)
        x7 = self.relu7(x7)

        x7 = self.aspp(x7)
        #x7=NonLocalBlock(x7)
        #x7=self.nl(x7)


        up1 = self.seb1(x7, x6)
        #up2 = self.seb2(up1, x8)
        up2 = self.seb2(up1, x5)
        up3 = self.seb3(up2, x4)
        up4 = self.seb4(up3, x3)
        up5 = self.seb5(up4, x2)
        up6 = self.seb6(up5, x1)



        out = self.heatmap(up6)
        # out torch.Size([2, 1, 256, 256])
        return out

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 　ｘ卷积后shape发生改变,比如:x:[1,64,56,56] --> [1,128,28,28],则需要1x1卷积改变x
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None

    def forward(self, x):
        # print(x.shape)
        o1 = self.relu(self.bn1(self.conv1(x)))
        # print(o1.shape)
        o2 = self.bn2(self.conv2(o1))
        # print(o2.shape)

        if self.conv1x1:
            x = self.conv1x1(x)

        out = self.relu(o2 + x)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Residual(64, 64),
            Residual(64, 64),
            Residual(64, 64),
        )

        self.conv3 = nn.Sequential(
            Residual(64, 128, stride=2),
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128),
        )

        self.conv4 = nn.Sequential(
            Residual(128, 256, stride=2),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
        )

        self.conv5 = nn.Sequential(
            Residual(256, 512, stride=2),
            Residual(512, 512),
            Residual(512, 512),
        )

        # self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 代替AvgPool2d以适应不同size的输入
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avg_pool(out)
        out = out.view((x.shape[0], -1))

        out = self.fc(out)

        return out


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False)
    )
    return layer
class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x
def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer

class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out





if __name__ == "__main__":
    model = KeyPointModel()




    x = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, inputs=(x,))
    print(flops)
