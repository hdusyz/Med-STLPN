import typing
import torch
import torch.nn as nn


def make_conv3d(in_channels: int, out_channels: int, kernel_size: typing.Union[int, tuple], stride: int,
                padding: int, dilation=1, groups=1,
                bias=True) -> nn.Module:
    """
    produce a Conv3D with Batch Normalization and ReLU

    :param in_channels: num of in in
    :param out_channels: num of out channels
    :param kernel_size: size of kernel int or tuple
    :param stride: num of stride
    :param padding: num of padding
    :param bias: bias
    :param groups: groups
    :param dilation: dilation
    :return: my conv3d module
    """
    module = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=groups,
                  bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.ReLU())
    return module


def conv3d_same_size(in_channels, out_channels, kernel_size, stride=1,
                     dilation=1, groups=1,
                     bias=True):
    padding = kernel_size // 2
    return make_conv3d(in_channels, out_channels, kernel_size, stride,
                       padding, dilation, groups,
                       bias)


def conv3d_pooling(in_channels, kernel_size, stride=1,
                   dilation=1, groups=1,
                   bias=False):
    padding = kernel_size // 2
    return make_conv3d(in_channels, in_channels, kernel_size, stride,
                       padding, dilation, groups,
                       bias)


class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // 4, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // 4, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention3D, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class LPA3D(nn.Module):
    def __init__(self, in_channel):
        super(LPA3D, self).__init__()
        self.ca = ChannelAttention3D(in_channel)
        self.sa = SpatialAttention3D()

    def forward(self, x):
        x0, x1 = x.chunk(2, dim=2)
        x0 = x0.chunk(2, dim=3)
        x1 = x1.chunk(2, dim=3)
        x0 = [self.ca(x0[-2]) * x0[-2], self.ca(x0[-1]) * x0[-1]]
        x0 = [self.sa(x0[-2]) * x0[-2], self.sa(x0[-1]) * x0[-1]]

        x1 = [self.ca(x1[-2]) * x1[-2], self.ca(x1[-1]) * x1[-1]]
        x1 = [self.sa(x1[-2]) * x1[-2], self.sa(x1[-1]) * x1[-1]]

        x0 = torch.cat(x0, dim=3)
        x1 = torch.cat(x1, dim=3)
        x3 = torch.cat((x0, x1), dim=2)

        x4 = self.ca(x) * x
        x4 = self.sa(x4) * x4
        x = x3 + x4
        return x

class TransformerAttention3D(nn.Module):
    def __init__(self, embed_dim, num_heads, spatial_dim, dropout=0.1):
        super(TransformerAttention3D, self).__init__()
        self.spatial_dim = spatial_dim

        # Multi-head Self-Attention
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Spatial-wise attention
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, seg):
        # Flatten spatial dimensions for MHA
        b, c, d, h, w = x.shape
        x_flattened = x.view(b, c, -1).permute(0, 2, 1)  # Shape: [b, d*h*w, c]

        # Add segmentation-guided spatial context
        avg_seg = torch.mean(seg, dim=1, keepdim=True)
        max_seg, _ = torch.max(seg, dim=1, keepdim=True)
        seg_features = torch.cat([avg_seg, max_seg], dim=1)  # Shape: [b, 2, d, h, w]
        spatial_weights = self.spatial_attn(seg_features)   # Shape: [b, 1, d, h, w]
        x = x * spatial_weights  # Weighted input features

        # Apply MHA
        attn_output, _ = self.mha(x_flattened, x_flattened, x_flattened)
        x = self.norm1(x_flattened + attn_output)

        # Apply FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # Shape: [b, d*h*w, c]

        # Reshape back to original spatial dimensions
        x = x.permute(0, 2, 1).view(b, c, d, h, w)

        return x


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        self.my_conv1 = make_conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.my_conv2 = make_conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = make_conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        out1 = self.conv3(inputs)
        out = self.my_conv1(inputs)
        out = self.my_conv2(out)
        out = out + out1
        return out


class ConvResLPA3D(nn.Module):
    def __init__(self, config):
        super(ConvResLPA3D, self).__init__()
        self.conv1 = conv3d_same_size(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = conv3d_same_size(in_channels=4, out_channels=4, kernel_size=3)
        self.config = config
        self.last_channel = 4
        self.first_lpa = LPA3D(4)
        layers = []
        i = 0
        for stage in config:
            i += 1
            layers.append(conv3d_pooling(self.last_channel, kernel_size=3, stride=2))
            for channel in stage:
                layers.append(ResidualBlock3D(self.last_channel, channel))
                self.last_channel = channel
            layers.append(LPA3D(self.last_channel))
        self.layers = nn.Sequential(*layers)
        self.attention = TransformerAttention3D(embed_dim=self.last_channel, num_heads=4, spatial_dim=(64, 64))
        self.avg_pooling = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_features=512, out_features=2)

    def forward(self, inputs, seg):
        out = self.conv1(inputs)
        out = self.conv2(out)
        #print(out.shape)
        out = self.first_lpa(out)
        out = self.layers(out)
        out1 = self.conv1(seg)
        out1 = self.conv2(out1)
        out1 = self.first_lpa(out1)
        out1 = self.layers(out1)
        out = self.attention(out, out1)
        # Classification head
        out = self.avg_pooling(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    
if __name__ == '__main__':
    model = ConvResLPA3D([[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]]) 
    x = torch.randn(1, 1, 16, 512, 512)
    seg = torch.randn(1, 1, 16, 512, 512)
    print(model(x,seg).size())
