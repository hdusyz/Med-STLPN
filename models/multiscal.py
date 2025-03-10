import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    """3D ResNet 残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.3):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)  # 加入 Dropout
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):
    """3D ResNet 构架"""
    def __init__(self, block, layers, in_channels=1, dropout_rate=0.3):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_rate=dropout_rate)

    def _make_layer(self, block, out_channels, blocks, stride, dropout_rate):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_rate))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, dropout_rate=dropout_rate))
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

        return x

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention

class MultiScaleResNet3D(nn.Module):
    """多尺度基于 ResNet 的特征提取模型"""
    def __init__(self, dropout_rate=0.3):
        super(MultiScaleResNet3D, self).__init__()
        # 全局和ROI特征提取模块
        self.global_extractor = ResNet3D(BasicBlock3D, [3, 4, 6, 3], dropout_rate=dropout_rate)
        self.roi_extractor = ResNet3D(BasicBlock3D, [3, 4, 6, 3], dropout_rate=dropout_rate)

        # 空间注意力模块
        self.spatial_attention = SpatialAttention()

        # 融合层
        self.fusion_fc = nn.Linear(512 * 2, 512)

        # 分类器
        self.classifier = nn.Linear(512, 2)  # 二分类任务

    def forward(self, global_input, roi_input):
        # 全局特征提取
        global_features = self.global_extractor(global_input)  # [B, 512, D', H', W']
        global_features = self.spatial_attention(global_features)  # 空间注意力增强
        global_features = F.adaptive_avg_pool3d(global_features, (1, 1, 1)).view(global_features.size(0), -1)

        # 局部特征提取
        roi_features = self.roi_extractor(roi_input)  # [B, 512, D'', H'', W'']
        roi_features = F.adaptive_avg_pool3d(roi_features, (1, 1, 1)).view(roi_features.size(0), -1)

        # 特征融合
        fused_features = torch.cat([global_features, roi_features], dim=1)  # [B, 512 * 2]
        fused_features = self.fusion_fc(fused_features)  # 降维到 512

        # 分类输出
        output = self.classifier(fused_features)  # [B, 2]
        return output

if __name__ == "__main__":
# 模拟输入数据
    global_input = torch.randn(4, 1, 16, 512, 512)  # 全图输入
    roi_input = torch.randn(4, 1, 16, 64, 64)      # ROI输入

# 实例化模型
    model = MultiScaleResNet3D(dropout_rate=0.3)
    output = model(global_input, roi_input)
    print("Output shape:", output.shape)  # 输出形状 [B, 2]