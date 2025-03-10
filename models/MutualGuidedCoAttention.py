import torch
import torch.nn as nn
import torch.nn.functional as F


    
class MutualGuidedCoAttention(nn.Module):
    def __init__(self, img_dim, text_dim, attention_dim):
        super(MutualGuidedCoAttention, self).__init__()
        
        # 定义线性层，将输入映射到共同的注意力空间
        self.query_linear = nn.Linear(img_dim, attention_dim)  # Q 图像 -> 线性层
        self.key_linear = nn.Linear(text_dim, attention_dim)    # K 文本-> 线性层
        self.value1 = nn.Linear(img_dim, attention_dim)  # V1 -> 图像的线性层
        self.value2 = nn.Linear(text_dim, attention_dim)  # V2 -> 文本的线性层
        
       # self.scale = attention_dim ** 0.5  # 缩放因子

    def forward(self, img_feature, text_embed):
        # 计算 Q, K, V1, V2
        Q = self.query_linear(img_feature)    # 图像输出作为 Query
        K = self.key_linear(text_embed)     # 文本嵌入作为 Key
        V1 = self.value1(img_feature)  # 图像嵌入作为 Value1
        V2 = self.value2(text_embed)   # 文本输出作为 Value2

        # 计算注意力分数
        attn_score_fusion = torch.matmul(Q, K.transpose(-2, -1))  # Q 和 K 计算点积
        attn_weights_text = torch.matmul(attn_score_fusion.transpose(-2,-1), V2)  # Q 和 V2 计算点积
        
        # 用文本的注意力权重来加权 V1 (文本)
        attended_text = attn_weights_text + V2
        
        # 用相同的机制处理 V2
        attn_weights_img = torch.matmul(attn_score_fusion.transpose(-2,-1), V1)  # Q 和 V1 计算点积
        
        # 用 LSTM 的注意力权重加权 V2 (LSTM)
        attended_img = attn_weights_img + V1

        return attended_img, attended_text

class OnlineDataAugmentation(nn.Module):
    def __init__(self, d_model):
        super(OnlineDataAugmentation, self).__init__()
        
        # Matching Layer
        self.matching_layer = nn.Linear(d_model, d_model)
        
        # Fusing Layer
        self.fusing_layer = nn.Linear(d_model * 2, d_model)
        
        # Scoring Layer (for ranking or scoring)
        self.scoring_layer = nn.Linear(d_model, 1)
        
        # FFN (Feed Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # 扩展维度
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)  # 投影回原始维度
        )
        
        # LayerNorm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x1):
        fused_features = self.matching_layer(x1)  # [batch_size, d_model]
        
        # 3. FFN transformation (Feed Forward Network)
        ffn_output = self.ffn(fused_features)  # [batch_size, d_model]
        
        # 4. Add & Norm operation
        output = self.norm(fused_features + ffn_output)  # [batch_size, d_model]
        
        
        return output
    
class SNNMixer(nn.Module):
    def __init__(self, d_model):
        super(SNNMixer, self).__init__()
        self.snn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
        self.snn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, x):
        x = self.snn1(x)
        x = self.snn2(x)
        return x

class MultimodalModel(nn.Module):
    def __init__(self, img_feat_dim=128, text_feat_dim=128, fusion_dim=128):
        super(MultimodalModel, self).__init__()

        # 图像特征线性投影到融合维度
        self.img_proj = nn.Linear(img_feat_dim, fusion_dim)
        
        # 文本特征线性投影到融合维度
        self.text_proj = nn.Linear(text_feat_dim, fusion_dim)

        # 互导注意力模块
        self.co_attention = MutualGuidedCoAttention(img_dim=128, text_dim=128, attention_dim=128)

        # 在线数据增强
        self.data_augmentation = OnlineDataAugmentation(fusion_dim)

        # SNN-Mixer模块
        self.snn_mixer = SNNMixer(fusion_dim)

        # FFN和归一化层
        self.ffn = nn.Linear(fusion_dim*2, fusion_dim)
        self.norm = nn.ReLU()

        # 最后一层用于二分类
        self.classifier = nn.Linear(fusion_dim, 2)  # 如果使用Sigmoid，则输出1个节点

    def forward(self, img_feat, text_feat):
        # 输入形状假设为 [batch_size, img_feat_dim] 和 [batch_size, text_feat_dim]
        

        # 对齐的图像特征和文本特征经过 Co-Attention
        co_attended_img_feat, co_attended_text_feat= self.co_attention(img_feat, text_feat)
        fusion = torch.cat([co_attended_img_feat, co_attended_text_feat], dim=1)
        out = self.ffn(fusion)
        out = self.norm(out)
        out = self.classifier(out)


        # 数据增强，融合图像和文本特征
        #fused_features = self.data_augmentation(co_attended_img_feat)

        # 通过SNN-Mixer进一步处理
        #snn_features = self.snn_mixer(co_attended_text_feat)
        # **特征拼接**：将来自 `OnlineDataAugmentation` 和 `SNNMixer` 的特征拼接
        #concatenated_features = torch.cat((fused_features, snn_features), dim=-1)  # 在最后一个维度上拼接，输出形状为 [batch_size, 2*fusion_dim]

        # FFN和LayerNorm
        #output = self.ffn(concatenated_features)
        #output = self.norm(output)
       # print(output.shape)
        # 最后一层线性层用于二分类
        #logits = self.classifier(output)  # [batch_size, 1]
       # print(logits.shape)
        return out

def test():
    global debug
    debug = True
    net = MultimodalModel(img_feat_dim=512, text_feat_dim=128, fusion_dim=512, num_heads=8)
    img_feat = torch.randn(4,1,512)  # 图像特征
    text_feat = torch.randn(4,1, 128)  # 文本特征
# 模型前向传播
    output = net(img_feat, text_feat)
    print(output.shape)
    
if __name__ == '__main__':
    test()
    
    


