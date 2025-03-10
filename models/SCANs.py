import torch
import torch.nn as nn
import torch.nn.functional as F

class SliceAttentionModel(nn.Module):
    def __init__(self):
        super(SliceAttentionModel, self).__init__()
        
        # 2D CNN for feature extraction (shared weights for each slice)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # 1D Conv layer for slice-wise attention
        self.attention_conv = nn.Conv1d(512, 512, kernel_size=1)  # Applying attention along the depth axis
        
        # Fully connected layers for final classification
        self.fc1 = nn.Linear(512 * 4 * 4 * 2, 128)  # Adjust input size based on final feature map size
        
    def forward(self, x):
        batch_size, depth, h, w = x.size()  # Input shape: (batch_size, depth, height, width)
        
        # Reshape for 2D CNN: (batch_size*depth, 1, height, width)
        x = x.view(-1, 1, h, w)
        
        # 2D CNN feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Reshape back to (batch_size, depth, channels, height, width)
        x = x.view(batch_size, depth, 512, x.size(-2), x.size(-1))  # Shape: (batch_size, depth, 512, 4, 4)
        
        # Flatten spatial dimensions: (batch_size, depth, channels, h*w)
        x = x.view(batch_size, depth, 512, -1)  # Shape: (batch_size, depth, 512, 16)
        
        # Now transpose for attention_conv to apply along the depth axis: (batch_size, 512, depth)
        x_transposed = x.permute(0, 2, 1, 3)  # Shape: (batch_size, 512, depth, h*w)
        
        # Apply attention mechanism along the depth dimension
        attention_weights = torch.sigmoid(self.attention_conv(x_transposed.mean(dim=-1)))  # Shape: (batch_size, 512, depth)
        
        # Apply attention to the feature maps (element-wise multiplication)
        attended_features = attention_weights.unsqueeze(-1) * x_transposed  # Shape: (batch_size, 512, depth, h*w)
        
        # Concatenate original features with attended features for each slice
        concatenated_features = torch.cat((x_transposed, attended_features), dim=-1)  # Shape: (batch_size, 512, depth, 2 * h*w)
        
        # Sum over depth dimension to merge slice-wise features
        combined_features = concatenated_features.sum(dim=2)  # Shape: (batch_size, 512, 2 * h*w)
        
        # Flatten the features for classification
        combined_features = combined_features.view(batch_size, -1)  # Shape: (batch_size, 512 * 2 * h * w)
        
        return combined_features

class TwoTimePointModel(nn.Module):
    def __init__(self):
        super(TwoTimePointModel, self).__init__()
        
        # Slice attention model for each time point
        self.slice_attention_model = SliceAttentionModel()
        
        # Fully connected layers for classification after concatenation of two time points' features
        self.fc1 = nn.Linear(512 * 4 * 4 * 4, 128)  # Adjust input size based on final feature map size
        self.fc2 = nn.Linear(128, 2)  # Binary classification
        
    def forward(self, x1, x2):
        # Pass the first time point through the slice attention model
        features_t1 = self.slice_attention_model(x1)
        
        # Pass the second time point through the slice attention model
        features_t2 = self.slice_attention_model(x2)
        
        # Concatenate features from both time points
        combined_features = torch.cat((features_t1, features_t2), dim=1)  # Concatenate along feature dimension
        
        # Fully connected layers for classification
        x = F.relu(self.fc1(combined_features))
        x = self.fc2(x)
        
        return x
# Example usage
#model = TwoTimePointModel()
#input_data_t1 = torch.randn(4, 16, 64, 64)  # batch_size=4, 16 slices, each 64x64 for time point 1
#input_data_t2 = torch.randn(4, 16, 64, 64)  # batch_size=4, 16 slices, each 64x64 for time point 2
#output = model(input_data_t1, input_data_t2)
#print(output.shape)  # Output should be [4, 2] for binary classification