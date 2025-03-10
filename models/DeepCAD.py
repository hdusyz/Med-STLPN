import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add shortcut connection
        out = F.relu(out)
        return out

class ComplexEncoder(nn.Module):
    def __init__(self):
        super(ComplexEncoder, self).__init__()
        # First convolution layer before entering residual blocks
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)  # 1*16*64*64 -> 64*8*32*32
        self.bn1 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)  # 64*8*32*32 -> 64*4*16*16
        
        # Residual blocks
        self.res_block1 = ResidualBlock(64, 128, stride=2)  # 64*4*16*16 -> 128*2*8*8
        self.res_block2 = ResidualBlock(128, 256, stride=2)  # 128*2*8*8 -> 256*1*4*4
        
        # Additional convolution layer to further reduce spatial dimension
        self.conv2 = nn.Conv3d(256, 512, kernel_size=3, padding=1)  # 256*1*4*4 -> 512*1*4*4
        self.bn2 = nn.BatchNorm3d(512)
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout3d(0.3)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Initial conv + pooling
        x = self.res_block1(x)  # First residual block
        x = self.res_block2(x)  # Second residual block
        x = F.relu(self.bn2(self.conv2(x)))  # Additional conv layer
        x = self.dropout(x)  # Dropout layer
        return x  # Output shape: batch_size * 512 * 1 * 4 * 4

class MetadataEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MetadataEncoder, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        return F.relu(self.fc(x))  # 输出文本特征

class LSTMClassifier(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM output
        lstm_out = lstm_out[:, -1, :]  # Use the output of the last LSTM cell
        return lstm_out

class DeepCAD(nn.Module):
    def __init__(self, categories, roi_feature_size=512*1*4*4, lstm_hidden_size=128, num_layers=2, num_special_tokens=2, metadata_hidden_size=64, num_classes=2):
        super(DeepCAD, self).__init__()
        # 选择复杂化的编码器：可以是 ComplexEncoder 或 ComplexDenseEncoder
        self.encoder = ComplexEncoder()  # 或者 self.encoder = ComplexDenseEncoder()
        self.lstm_classifier = LSTMClassifier(roi_feature_size, lstm_hidden_size, num_layers)
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens
        self.linear_proj = nn.Linear(512, 128)  # d_embed 为原始嵌入维度
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, 512)
        
        # 将LSTM输出和Metadata特征结合后通过全连接层
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, t0, t1, text_data):
        # 编码T0和T1
        t0_encoded = self.encoder(t0)
        t1_encoded = self.encoder(t1)
        
        # Flatten 3D features into 1D vectors
        t0_flatten = t0_encoded.view(t0_encoded.size(0), -1)
        t1_flatten = t1_encoded.view(t1_encoded.size(0), -1)
        
        # Stack T0 and T1 along the sequence dimension
        features = torch.stack([t0_flatten, t1_flatten], dim=1)
        
        # Pass through LSTM
        lstm_out = self.lstm_classifier(features)
        
        # 编码metadata
        assert text_data.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        x_categ = text_data + self.categories_offset
        x_categ = self.categorical_embeds(x_categ)
        x_categ = x_categ.squeeze(1)  # 去掉多余的维度 (B, num_categories, d_embed)
        x_categ = self.linear_proj(x_categ)  # (B, num_categories, 128)
        #print(x_categ.shape)
        x_categ = x_categ.mean(dim=1)  # 或者 x_categ.sum(dim=1) 根据需求
        # Combine LSTM output and metadata
        combined_features = torch.cat([lstm_out, x_categ], dim=1)
        
        # 最终分类
        output = self.fc(combined_features)
        return output