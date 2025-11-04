import os
import random
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset , dataloader
import torchio as tio
from torchio import Queue
from torchio import SubjectsDataset, UniformSampler
from torch.nn import functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchsummary import summary
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

class firstConv3d(nn.Module):
    """
    This block will help us to convert 4 modalities into
    32 feature set which is more broader than the 4  modalities
    so we take input of the those 4 modalities and output 32
    feature maps after applying 3x3x3 convolution,
    batch normalization and relu activation function
    """
    def __init__(self, input_size, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv3d(input_size, 32, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm3d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x



class downSampling(nn.Module):
    """
    This will be our encoder layer or downsampling block which will
    be responsible for converting 32 in channels to 512 out channels
    after applying 3x3x3 convolution, batch normalization and relu activation function
    and then maxpooling.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        return x, skip


class bottleNeck(nn.Module):
    """
    This will be out bottleneck layer where we apply the
    batchnormalization to the 512 features map two time for reguralization
    and then apply the relu activation function.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.dropout = nn.Dropout3d(p=0.3)  # Optional regularization

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x



# class upSampling(nn.Module):
#     """
#     In upSampling we do reverse Convolution Operation with the help
#     nn.ConvTranspose2d to mid channel and then concate that mid channel with the
#     output feature map of the corresponding encoder block and also increase the
#     padding of the skips is the size mismatch occur between concating operand an
#     skip with the help of F.pad(x , [left , right , top , botton])
#     """
#     def __init__(self, in_channels, mid_channels, out_channels):
#         super(upSampling, self).__init__()
#         # Upsample: reduce channels and double spatial size
#         self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)

#         # Double convolution after concatenation
#         self.conv1 = nn.Conv2d(mid_channels + out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#     def forward(self, x, skip):
#         x = self.up(x)  # Upsample
#         # Pad if needed to match skip connection size
#         if x.size() != skip.size():
#             diffY = skip.size()[2] - x.size()[2]
#             diffX = skip.size()[3] - x.size()[3]
#             x = F.pad(x, [diffX // 2, diffX - diffX // 2,
#                           diffY // 2, diffY - diffY // 2])
#         x = torch.cat([skip, x], dim=1)  # Concatenate along channel axis
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         return x

class AttentionGate(nn.Module):
    """Implemented the attention layer for filtering out unimportant features"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class  upSampling(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(F_g=out_channels, F_l=skip_channels, F_int=out_channels // 2)

        self.conv1 = nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Pad if needed
        if x.shape[2:] != skip.shape[2:]:
            diffZ = skip.size(2) - x.size(2)
            diffY = skip.size(3) - x.size(3)
            diffX = skip.size(4) - x.size(4)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2,
                          diffZ // 2, diffZ - diffZ // 2])

        # Apply attention gate to skip connection
        skip = self.attention(x, skip)

        # Concatenate and refine
        x = torch.cat([skip, x], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class uNET(nn.Module):
  def __init__(self , in_channel , out_channel ):
    super(uNET, self).__init__() # Added self here
    self.dovConv = firstConv3d(in_channel)

    # Encoder Blocks
    self.d1 = downSampling(32 , 64)
    self.d2 = downSampling(64 , 128)
    self.d3 = downSampling(128 , 256)
    self.d4 = downSampling(256 , 512)

    # BottleNeck Block
    self.b1 = bottleNeck(512 , 512)

    # Decoder Block
    self.u1 = upSampling(512, 512, 256)  # in, skip, out
    self.u2 = upSampling(256, 256, 128)
    self.u3 = upSampling(128, 128, 64)
    self.u4 = upSampling(64, 64, 32)


    # Final Output
    self.f = nn.Conv3d(32 , out_channel , kernel_size = 1 )

  def forward(self , x):
    x = self.dovConv(x)
    x1 , skip1 = self.d1(x)
    x2 , skip2 = self.d2(x1)
    x3 , skip3 = self.d3(x2)
    x4 , skip4 = self.d4(x3)

    x5 = self.b1(x4)

    x = self.u1(x5 , skip4)
    x = self.u2(x , skip3)
    x = self.u3(x , skip2)
    x = self.u4(x , skip1)

    out = self.f(x)

    return out