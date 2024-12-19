import torch 
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = self._conv_block(in_channels, in_channels * 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = self._conv_block(in_channels * 16, in_channels * 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = self._conv_block(in_channels * 32, in_channels * 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = self._conv_block(in_channels * 64, in_channels * 128, kernel_size=4, stride=2, padding=1)
        self.conv5 = self._conv_block(in_channels * 128, in_channels * 256, kernel_size=4, stride=2, padding=1)
        self.conv6 = self._conv_block(in_channels * 256, in_channels * 512, kernel_size=4, stride=2, padding=1)
        self.out_layer = self._conv_block(in_channels * 512, in_channels , kernel_size=4, stride=2, padding=1, activation = False)

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding, activation = True):
        if activation:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
                nn.BatchNorm2d(out_channels,affine=True),
                nn.LeakyReLU(0.2)
                )
        else:
            return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.out_layer(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, channels, filters):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = self._conv_blockT(in_channels = latent_dim, out_channels = filters * 128, kernel_size = 4,stride = 2, padding = 1)
        self.conv2 = self._conv_blockT(in_channels = filters * 128, out_channels = filters * 64, kernel_size = 4,stride = 2, padding = 1)
        self.conv3 = self._conv_blockT(in_channels = filters * 64, out_channels = filters * 32, kernel_size = 4,stride = 2, padding = 1)
        self.conv4 = self._conv_blockT(in_channels = filters * 32 , out_channels = filters * 16,kernel_size = 4,stride = 2,padding = 1)
        self.conv5 = self._conv_blockT(in_channels = filters * 48 , out_channels = filters * 16,kernel_size = 4,stride = 2,padding = 1)
        self.conv6 = self._conv_blockT(in_channels = filters * 16, out_channels = filters * 6,kernel_size = 4,stride = 2,padding = 1)
        self.conv7 = self._conv_blockT(in_channels = filters * 6, out_channels = filters * 4,kernel_size = 4,stride = 2,padding = 1)
        self.conv8 = self._conv_blockT(in_channels = filters * 6, out_channels = filters * 3,kernel_size = 4,stride = 2,padding = 1)

        self.out_layer = nn.Sequential(
            nn.Conv2d(in_channels = filters * 3, out_channels = channels, kernel_size = 4, stride = 2, padding = 1)
            )

        self.skip2 = self._conv_blockT(in_channels = filters * 64, out_channels = filters * 32, kernel_size = 7,stride = 3, padding = 0)
        self.skip3 = self._conv_blockT(in_channels = filters * 16, out_channels = filters * 2, kernel_size = 8,stride = 4, padding = 2)

    def _conv_blockT(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        y = self.skip2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.cat([x,y],dim=1)
        x = self.conv5(x)
        y = self.skip3(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.cat([x,y],dim=1)
        x = self.conv8(x)
        x = self.out_layer(x)
        return torch.tanh(x)

x = torch.randn(1,3,128,128)
d = Discriminator(3)
assert d(x).shape == torch.randn(1,3,1,1).shape, "Incorrect Discriminator Shape!"
g = Generator(128,3,8)
z = torch.randn(1,128,1,1)
assert g(z).shape == x.shape, "Incorrect Generator Shape!"