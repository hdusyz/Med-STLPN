import torch
import torch.nn as nn
import torch.nn.functional as F

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(),
            nn.Linear(in_channels // 16, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.global_avg_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        avg_out = self.fc(avg_out)
        avg_out = avg_out.view(avg_out.size(0), avg_out.size(1), 1, 1, 1)
        return x * self.sigmoid(avg_out)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) 
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        attention_map = torch.cat([avg_out, max_out], dim=1)  
        attention_map = self.conv(attention_map) 
        return x * self.sigmoid(attention_map) 


# Group Attention Block
class GroupAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=4):
        super(GroupAttentionBlock, self).__init__()
        self.num_groups = num_groups
        self.group_conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.group_conv3x3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)

    def forward(self, x):
        group1_out = self.group_conv1x1(x)
        group2_out = self.group_conv3x3(x)
        out = group1_out + group2_out
        out = self.channel_attention(out)
        out = self.spatial_attention(out)
        return out

# ResGANet块
class ResGANetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=4, stride=1):
        super(ResGANetBlock, self).__init__()
        
        self.group_attention = GroupAttentionBlock(in_channels, out_channels, num_groups)

        if in_channels != out_channels:
            self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x  
        out = self.group_attention(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class ResGANet101(nn.Module):
    def __init__(self, num_classes=2):
        super(ResGANet101, self).__init__()
        
        # 第一层卷积，批归一化和ReLU
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 处理输入通道数为1
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 使用 ResGANetBlock 替代瓶颈块
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 23, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # 最后的全连接层
        self.avgpool1 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.avgpool2 = nn.AdaptiveAvgPool3d((16, 512, 512))

        self.channel_align = nn.Conv3d(512, 1, kernel_size=1, stride=1, padding=0)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResGANetBlock(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(ResGANetBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x_flag = x
        x_flag = self.avgpool2(x_flag)
        x_flag = self.channel_align(x_flag)

        x = self.avgpool1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x_flag, x

    
# # 测试
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ResGANet101().to(device)
# x = torch.randn(1, 1, 16, 512, 512).to(device)
# output, _ = model(x)
# print(output.size(), _.size())
