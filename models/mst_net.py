import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.camf import *

class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=(2,4,4), stride=(2,4,4), padding=0, output_padding=0)

    def forward(self, x):
        #print('SingleDeconv3DBlock:',self.block(x).shape)
        return self.block(x)

class SingleDeconv3DBlock1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        #print('SingleDeconv3DBlock:',self.block(x).shape)
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        #print('SingleConv3DBlock:',self.block(x).shape)
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class DynamicReceptiveFieldBlock(nn.Module):
    """四尺度增强动态感受野模块"""
    def __init__(self, in_ch):
        super().__init__()
        # 扩张率对应器官、病灶、组织、细胞四个尺度
        self.conv1 = nn.Conv3d(in_ch, in_ch//4, 3, dilation=1, padding=1)  # 细胞级特征
        self.conv2 = nn.Conv3d(in_ch, in_ch//4, 3, dilation=2, padding=2)  # 组织级特征
        self.conv3 = nn.Conv3d(in_ch, in_ch//4, 3, dilation=4, padding=4)  # 病灶级特征
        self.conv4 = nn.Conv3d(in_ch, in_ch//4, 3, dilation=8, padding=8)  # 器官级特征
        
        # 四分支自适应权重
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),      # 全局特征描述符
            nn.Conv3d(in_ch, 4, 1),       # 生成各分支权重
            nn.Softmax(dim=1)             # 权重归一化
        )

    def forward(self, x):
        # 并行四尺度特征提取 (核心改进点)
        w = self.attn(x)  # 获取自适应权重[B,4,1,1,1]
        
        # 权重分配（扩展为四个维度）
        x1 = self.conv1(x) * w[:, 0:1]  # 微尺度特征强化
        x2 = self.conv2(x) * w[:, 1:2]  # 细观特征强化
        x3 = self.conv3(x) * w[:, 2:3]  # 亚病灶级特征强化
        x4 = self.conv4(x) * w[:, 3:4]  # 全景特征强化
        
        # 特征融合 (关键改变点)
        return torch.cat([x1, x2, x3, x4], dim=1)  # 维度对齐确保输出通道与输入相同


class STDM(nn.Module):
    """3D时空特征建模"""
    def __init__(self, in_ch):
        super().__init__()
        
        # 轴分离卷积
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, (1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(in_ch),
            nn.GELU()
        )
        
        # 双向长程时序建模
        self.temporal_attn = nn.MultiheadAttention(embed_dim=in_ch, num_heads=2, batch_first=True)
        
    def forward(self, x):
        B,C,D,H,W = x.shape
        
        # 空间增强
        spatial_feat = self.spatial_conv(x)
        
        # 时序注意力
        temporal_feat = spatial_feat.permute(0,3,4,1,2)  # [B, H, W, C, D]
        temporal_feat = temporal_feat.reshape(B*H*W, C, D)  # [B*H*W, C, D]
        temporal_feat = temporal_feat.permute(0,2,1)  # [B*H*W, D, C]
        
        # 执行注意力计算（时间轴序列）
        temporal_feat, _ = self.temporal_attn(
            temporal_feat, temporal_feat, temporal_feat
        )  # [B*H*W, D, C]  
        
        # Step3. 恢复原始维度
        temporal_feat = temporal_feat.permute(0,2,1)  # [B*H*W, C, D]
        temporal_feat = temporal_feat.view(B, H, W, C, D)  # [B, H, W, C, D]
        temporal_feat = temporal_feat.permute(0,3,4,1,2)  # [B, C, D, H, W]
        
        return temporal_feat + spatial_feat

class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        
        mixed_query_layer = self.query(hidden_states)
        #print('mixq:',mixed_query_layer.shape)
        mixed_key_layer = self.key(hidden_states)
        #print('mixk:',mixed_key_layer.shape)
        mixed_value_layer = self.value(hidden_states)
       #print('mixv:',mixed_value_layer.shape)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        #print('query_layer:',query_layer.shape)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        #print('key_layer:',key_layer.shape)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #print('value_layer:',value_layer.shape)
        #print('key_layer.transpose(-1, -2):',key_layer.transpose(-1, -2).shape)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        #print('attention_scores:',attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        #print('context_layer:',context_layer.shape)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        #print('attention_output:',attention_output.shape)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    """
    3D ResNet 的基本残差块，包含两个 3D 卷积层。
    """
    expansion = 1  # 对于 BasicBlock，通道数不扩展

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        """
        Args:
            in_planes: 输入通道数
            planes: 输出通道数
            stride: 卷积步长，既可以是整数，也可以是元组，默认 1
            downsample: 是否需要下采样（用于匹配残差分支的尺寸），一般在改变通道数或下采样时使用
        """
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要，则对 shortcut 分支进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet3DFeatureExtractor(nn.Module):
    """
    自定义实现的 3D ResNet 特征提取器：
      - stem 部分：包含一个初始 3D 卷积、BatchNorm、ReLU 和 3D 最大池化
      - layer1、layer2、layer3：由 BasicBlock3D 组成，其中 layer2 和 layer3 首个 block 使用 stride=2 下采样
      - conv_expand：1×1×1 卷积将通道数从 256 扩展到 768
      - adaptive_pool：自适应平均池化到 (1, 4, 4)，即时间轴降为 1，空间为 4×4
    输入形状：(B, 1, 16, 512, 512)  → 输出形状：(B, 768, 1, 4, 4)
    """
    def __init__(self, block=BasicBlock3D, layers=[2, 2, 2]):
        super(ResNet3DFeatureExtractor, self).__init__()
        self.in_planes = 64
        # stem：初始卷积，将输入通道从 1 升到 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 7, 7),
                               stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        # 使用 3D 最大池化（只对空间下采样，时间保持不变）
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                    padding=(0, 1, 1))
        
        # layer1：不进行下采样，输出通道数保持 64
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        # layer2：下采样（stride=2），通道数由 64 扩展到 128
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # layer3：下采样（stride=2），通道数由 128 扩展到 256
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        # 使用 1×1×1 卷积将通道数从 256 扩展到 768
        self.conv_expand = nn.Conv3d(256 * block.expansion, 768, kernel_size=1, bias=False)
        # 自适应平均池化：将时间轴池化到 1，空间池化到 4×4
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 4, 4))

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建由多个残差块组成的层。
        Args:
            block: 残差块类型（这里使用 BasicBlock3D）
            planes: 输出通道数
            blocks: 残差块的个数
            stride: 第一层的步长（用于下采样）
        """
        downsample = None
        # 如果步长不为 1 或者通道数不匹配，则需要对 shortcut 分支进行下采样
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        # 第一个 block 可能需要下采样
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        # 后续的 block 不需要下采样，步长为 1
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播过程：
          输入 x: (B, 1, 16, 512, 512)
          conv1 + bn + relu: (B, 64, 16, 256, 256)
          maxpool: (B, 64, 16, 128, 128)
          layer1: (B, 64, 16, 128, 128)
          layer2: (B, 128, 8, 64, 64)   -- 下采样，时间从 16->8, 空间从 128->64
          layer3: (B, 256, 4, 32, 32)   -- 下采样，时间从 8->4, 空间从 64->32
          conv_expand: (B, 768, 4, 32, 32)
          adaptive_pool: (B, 768, 1, 4, 4)
        """
        x = self.conv1(x)    # (B, 64, 16, 256, 256)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (B, 64, 16, 128, 128)

        x = self.layer1(x)   # (B, 64, 16, 128, 128)
        x = self.layer2(x)   # (B, 128, 8, 64, 64)
        x = self.layer3(x)   # (B, 256, 4, 32, 32)

        x = self.conv_expand(x)   # (B, 768, 4, 32, 32)
        x = self.adaptive_pool(x) # (B, 768, 1, 4, 4)
        return x



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = (cube_size[0] // patch_size[0]) * (cube_size[1] // patch_size[1]) * (cube_size[2] // patch_size[2])
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print('x shape',x.shape)
        x = self.patch_embeddings(x)
        #print('x_patchembedding:',x.shape)
        x = x.flatten(2)
        #print('x_flatten:',x.shape)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        #print('embedding_shape:',embeddings.shape)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = (cube_size[0] // patch_size[0]) * (cube_size[1] // patch_size[1]) * (cube_size[2] // patch_size[2])
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        #print('Trans_x:',x.shape)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        #print('transblock_final_x:',x.shape)
        return x, weights


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, cube_size, patch_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)
        
        return extract_layers

    
class SpatioTemporalFusion(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        """
        d_model: 每个 token 的特征维度
        hidden_dim: MLP 中间层维度，默认为 d_model 的一半
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model // 2
        
        # 使用 LayerNorm 对输入特征做归一化
        self.norm = nn.LayerNorm(3 * d_model)
        # 轻量级 MLP，用于融合 t0、t1 以及它们的差分信息
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, t0_feat, t1_feat):
        """
        输入:
            t0_feat, t1_feat: [B, N, d_model] —— 分别为 t0 和 t1 的影像特征
        输出:
            融合后的特征: [B, N, d_model]
        """
        # 计算变化信息（差分）
        diff = t1_feat - t0_feat
        # 拼接 t0、t1 和变化信息，形状为 [B, N, 3*d_model]
        fusion_input = torch.cat([t0_feat, t1_feat, diff], dim=-1)
        # 对拼接结果进行归一化处理
        fusion_input = self.norm(fusion_input)
        # 通过轻量 MLP 获得融合特征
        fusion_output = self.mlp(fusion_input)
        # 使用残差连接：可以将 t0 和 t1 的平均值作为基本信息加到融合结果中
        fused = (t0_feat + t1_feat) / 2 + fusion_output
        return fused

    
class MST_NET_seg(nn.Module):
    def __init__(self,categories, img_shape=(16, 512, 512), input_dim=1, output_dim=1, embed_dim=768, patch_size=(16,128,128), num_heads=4, dropout=0.1, num_special_tokens=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        #text pre
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, 512)

        #self.patch_dim = [int(x / patch_size) for x in img_shape]
        self.patch_dim = [
            img_shape[0] // patch_size[0],
            img_shape[1] // patch_size[1],
            img_shape[2] // patch_size[2]
        ]

        # Transformer Encoder
        self.transformer = \
            Transformer(
                input_dim,
                embed_dim,
                img_shape,
                patch_size,
                num_heads,
                self.num_layers,
                dropout,
                self.ext_layers
            )

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(input_dim, 32, 3),
                Conv3DBlock(32, 64, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
                Deconv3DBlock(256, 128)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
            )

        self.decoder9 = \
            Deconv3DBlock(embed_dim, 512)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(embed_dim, 512)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(1024, 512),
                Conv3DBlock(512, 512),
                Conv3DBlock(512, 512),
                SingleDeconv3DBlock(512, 256)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(512, 256),
                Conv3DBlock(256, 256),
                SingleDeconv3DBlock(256, 128)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(256, 128),
                Conv3DBlock(128, 128),
                SingleDeconv3DBlock1(128, 64)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                SingleConv3DBlock(64, output_dim, 1)
            )
        
        self.avg_pool = nn.AvgPool3d(kernel_size=(1,4,4), stride=4)
        self.clshead = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )
        self.tepfusion = SpatioTemporalFusion(d_model=768)
        self.CMAF = CMFA(img_dim=256,tab_dim=256,hid_dim=256)
        self.linear_proj = nn.Linear(512, 256)  # d_embed 为原始嵌入维度
        self.stdm12 = STDM(768)
        self.stdm9 = STDM(512)
        self.stdm6 = STDM(256)
        self.stdm3 = STDM(128)
        self.stdm0 = STDM(64)
        self.jiangwei = nn.Sequential(
            nn.Linear(768,512),
            nn.ReLU(),
            nn.Linear(512,256),
        )

        self.clshead1 = nn.Sequential(
            nn.Linear(768,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )
    def forward(self, t0, t1, text_data):
        # deal with table data
        assert text_data.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        x_categ = text_data + self.categories_offset
        #print(x_categ.shape)
        x_categ = self.categorical_embeds(x_categ)
        x_categ = x_categ.squeeze(1)  # 去掉多余的维度 (B, num_categories, d_embed)
        #print(x_categ.shape)
        x_categ = self.linear_proj(x_categ)  # (B, num_categories, 128)
        #print(x_categ.shape)
        x_categ = x_categ.mean(dim=1)  # 或者 x_categ.sum(dim=1) 根据需求
        #x_categ = text_data
        #T0 branch
        z_t0 = self.transformer(t0)
        z0_t0, z3_t0, z6_t0, z9_t0, z12_t0 = t0, *z_t0
        #print('z0 shape:',z0.shape)
        # #print('z3 shape:',z3.shape)
        # #print('z6 shape:',z6.shape)
        # #print('z9 shape:',z9.shape)
        # #print('z12 shape:',z12.shape)
        z3_t0 = z3_t0.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6_t0 = z6_t0.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9_t0 = z9_t0.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12_t0 = z12_t0.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        clsfeat_t0 = z12_t0
        z12_t0 = self.stdm12(z12_t0)
        z12_t0 = self.decoder12_upsampler(z12_t0)
        z9_t0 = self.decoder9(z9_t0)
        z9_t0 = self.stdm9(z9_t0)
        z9_t0 = self.decoder9_upsampler(torch.cat([z9_t0, z12_t0], dim=1))
        z6_t0 = self.decoder6(z6_t0)
        z6 = self.stdm6(z6)
        z6_t0 = self.decoder6_upsampler(torch.cat([z6_t0, z9_t0], dim=1))
        # #print(z6.shape)
        z3_t0 = self.decoder3(z3_t0)
        # #z3 = self.stdm3(z3)
        
        z3_t0 = self.decoder3_upsampler(torch.cat([z3_t0, z6_t0], dim=1))
        z0_t0 = self.decoder0(z0_t0)
        # #z0 = self.stdm0(z0)
        segout = self.decoder0_header(torch.cat([z0_t0, z3_t0], dim=1))

        #T1 branch
        z_t1 = self.transformer(t1)
        z0_t1, z3_t1, z6_t1, z9_t1, z12_t1 = t1, *z_t1
        z3_t1 = z3_t1.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6_t1 = z6_t1.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9_t1 = z9_t1.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12_t1 = z12_t1.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        clsfeat_t1 = z12_t1
        z12_t1 = self.stdm12(z12_t1)
        z12_t1 = self.decoder12_upsampler(z12_t1)
        z9_t1 = self.decoder9(z9_t1)
        z9_t1 = self.stdm9(z9_t1)
        z9_t1 = self.decoder9_upsampler(torch.cat([z9_t1, z12_t1], dim=1))
        z6_t1 = self.decoder6(z6_t1)
        #z6 = self.stdm6(z6)
        z6_t1 = self.decoder6_upsampler(torch.cat([z6_t1, z9_t1], dim=1))
        z3_t1 = self.decoder3(z3_t1)
        #z3 = self.stdm3(z3)
        z3_t1 = self.decoder3_upsampler(torch.cat([z3_t1, z6_t1], dim=1))
        z0_t1 = self.decoder0(z0_t1)
        #z0 = self.stdm0(z0)
        segout1 = self.decoder0_header(torch.cat([z0_t1, z3_t1], dim=1))
        clsfeat_t0 = clsfeat_t0.detach()
        clsfeat_t1 = clsfeat_t1.detach()

        #class
        out0 = self.avg_pool(clsfeat_t0)
        out1 = self.avg_pool(clsfeat_t1)
        out1 = out1.view(out1.size(0), -1)

        clsout = self.clshead1(out1)

        return segout, clsout


class MST_NET_cls(nn.Module):
    def __init__(self,categories,embed_dim=768, patch_size=(16,128,128),dropout=0.1, num_special_tokens=2):
        super().__init__()
        #text pre
        #self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, 512)
        self.globalfeat = ResNet3DFeatureExtractor()
        self.avg_pool = nn.AvgPool3d(kernel_size=(1,4,4), stride=4)
        self.clshead = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )
        self.tepfusion = SpatioTemporalFusion(d_model=1536)
        self.CMAF = CMFA(img_dim=256,tab_dim=256,hid_dim=256)
        self.linear_proj = nn.Linear(512, 256)  # d_embed 为原始嵌入维度
        self.jiangwei = nn.Sequential(
            nn.Linear(1536,512),
            nn.ReLU(),
            nn.Linear(512,256),
        )
        self.clshead1 = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )
    def forward(self, t0, t0_global, t1, t1_global, text_data):
        # deal with table data
        assert text_data.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        x_categ = text_data + self.categories_offset
        #print(x_categ.shape)
        x_categ = self.categorical_embeds(x_categ)
        x_categ = x_categ.squeeze(1)  # 去掉多余的维度 (B, num_categories, d_embed)
        #print(x_categ.shape)
        x_categ = self.linear_proj(x_categ)  # (B, num_categories, 128)
        #print(x_categ.shape)
        x_categ = x_categ.mean(dim=1)  # 或者 x_categ.sum(dim=1) 根据需求

        #class
        t0_g = self.globalfeat(t0_global)
        t1_g = self.globalfeat(t1_global)
        t0 = torch.cat((t0,t0_g),dim=1)
        t1 = torch.cat((t1,t1_g),dim=1)
        out0 = self.avg_pool(t0)
        out1 = self.avg_pool(t1)
        out0 = out0.view(out0.size(0), -1)
        out1 = out1.view(out1.size(0), -1)
        out0 = out0.unsqueeze(1)
        out1 = out1.unsqueeze(1)
        out = self.tepfusion(out0,out1)
        out = out.squeeze(1)
        out = self.jiangwei(out)
        fusion = self.CMAF(out, x_categ)

        clsout = self.clshead1(fusion)
        return clsout


if __name__ =='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = torch.randn((1,1,16,512,512)).to(device)
    x = torch.randn((1,1,16,512,512)).to(device)
    text = torch.randn((1,256)).to(device)
    model = mst_net_seg(categories=[0,2]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    p,o= model(y,x,text)
    print(p.shape,o.shape)
