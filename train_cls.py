import math
import os
import random
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import optim
from torch.autograd import Variable
from torchvision import transforms
from datas.dataset import *
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
from utils import DiceLoss, metric_seg, cmp_3
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from modeltest.egeunet import *
from tqdm import tqdm  # Import tqdm
from muti_scale.MLTN import MTLN3D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # 新增 auc 指标
from models.mst_net import *



# 数据路径
csv_path = '/home/ocean/sy/MICCAI/dataset/merged_file.csv'  # 替换为你的 CSV 文件的路径
data_dir = '/home/ocean/sy/MICCAI/newglobal/'  # 替换为你的 .npy 文件所在的目录路径
text_dir = '/home/ocean/sy/MICCAI/dataset/scale_information.csv'
seg_dir = '/home/ocean/sy/MICCAI/segroi1/'
csv_data = pd.read_csv(csv_path)
text_data = pd.read_csv(text_dir)
subject_ids = csv_data['Subject ID'].unique()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main():
    # 初始化数据集（保持不变）
    train_set = multitimedataset(csv_data, data_dir, seg_dir, text_data, normalize=True)
    cv = KFold(n_splits=4, random_state=42, shuffle=True)
    fold = 1
    num_epochs = 25

    # 仅保留分类指标
    tr_loss = []
    val_loss = []
    test_acc = []     
    test_pre = []     
    test_recall = []  
    test_f1 = []       

    with open("med_cls_cmaf.txt", "w") as file:
        file.write("Classification Training Results\n")
        file.write("=" * 50 + "\n")

        for train_idx, test_idx in cv.split(train_set):
            print(f"\nCross validation fold {fold}")

            train_loader = DataLoader(
                train_set, batch_size=1,
                sampler=SubsetRandomSampler(train_idx),
                num_workers=0
            )
            
            test_loader = DataLoader(
                train_set, batch_size=1,
                sampler=SubsetRandomSampler(test_idx),
                num_workers=0
            )

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            seg_model = mst_net_seg().to(device)
            state_dict = torch.load('/home/ocean/sy/MICCAI/Result/seg/seg_1_model.pth', map_location='cpu')
            seg_model.load_state_dict(state_dict)
            seg_model.eval()  # 固定为评估模式
            for param in seg_model.parameters():
                param.requires_grad = False


            #model = ConvRes([[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]]).to(device)
            cls_model = mst_net_cls(train_set.num_cat).to(device)
            
            # 修改训练函数返回参数
            train_loss_epoch, avg_val_loss, val_acc, val_pre, val_recall, val_f1 = train(
                seg_model, cls_model, train_loader, test_loader, fold, num_epochs, file
            )

            # 记录分类指标
            tr_loss.append(train_loss_epoch)
            val_loss.append(avg_val_loss)
            test_acc.append(val_acc)
            test_pre.append(val_pre)
            test_recall.append(val_recall)
            test_f1.append(val_f1)
            
            fold += 1
            torch.cuda.empty_cache()

        # 最终输出分类结果
        print('\n', '#' * 10, 'Final 5-Fold Results', '#' * 10)
        print('Average Train Loss:{:.4f}'.format(np.mean(tr_loss)))
        print('Average Val Loss:{:.4f}'.format(np.mean(val_loss)))
        print('\nClassification Results:')
        print('Accuracy:{:.2%}±{:.4f}'.format(np.mean(test_acc), np.std(test_acc)))
        print('Precision:{:.2%}±{:.4f}'.format(np.mean(test_pre), np.std(test_pre)))
        print('Recall:{:.2%}±{:.4f}'.format(np.mean(test_recall), np.std(test_recall)))
        print('F1 Score:{:.2%}±{:.4f}'.format(np.mean(test_f1), np.std(test_f1)))

        # 写入最终结果
        file.write("\nFinal Results\n")
        file.write("=" * 50 + "\n")
        file.write(f"Average Train Loss: {np.mean(tr_loss):.4f}\n")
        file.write(f"Average Val Loss: {np.mean(val_loss):.4f}\n")
        file.write(f"Accuracy: {np.mean(test_acc):.2%}±{np.std(test_acc):.4f}\n")
        file.write(f"Precision: {np.mean(test_pre):.2%}±{np.std(test_pre):.4f}\n")
        file.write(f"Recall: {np.mean(test_recall):.2%}±{np.std(test_recall):.4f}\n")
        file.write(f"F1 Score: {np.mean(test_f1):.2%}±{np.std(test_f1):.4f}\n")

class AvgMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(seg_model, cls_model, train_loader, test_loader, fold, num_epochs, file):
    # 仅使用分类损失
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(cls_model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * 0.9 + 0.1
    )
    
    train_loss_pic = []
    val_loss_pic = []
    best_acc = 0.0
    save_path = f'./Result/cls/Train_{fold}_model.pth'

    for epoch in range(num_epochs):
        cls_model.train()  # 分类模型训练
        train_loss_meter = AvgMeter()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            images1, images2, labels, table_info = batch['T0_image'], batch['T1_image'], batch['label'], batch['table_info']
            images1 = images1.unsqueeze(1).to(device)  # 假设只使用 T0 图像提取特征
            images2 = images2.unsqueeze(1).to(device)  
            labels = labels.long().to(device)
            table_info = table_info.to(device)
            
            optimizer.zero_grad()
            # 先用冻结的分割模型提取特征
            with torch.no_grad():
                # 要求 UNETR 的 forward 支持 return_clsfeat 参数
                y0, x, clsfeat1 = seg_model(images1, return_clsfeat=True)
                y1, x, clsfeat2 = seg_model(images2, return_clsfeat=True)
            
            # 将提取的特征和 table_info 送入新的分类模型（假设 cls_model 接收两个输入）
            class_logits = cls_model(clsfeat1,y0,clsfeat2,y1,table_info)
            loss = criterion(class_logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_meter.update(loss.item(), images1.size(0))
        
        # 验证部分
        val_loss, val_acc, val_pre, val_recall, val_f1 = validate(seg_model, cls_model, test_loader, file)
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(cls_model.state_dict(), save_path)
        
        train_loss_pic.append(train_loss_meter.avg)
        val_loss_pic.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss_meter.avg:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Classification -> Accuracy: {val_acc:.2%} | Precision: {val_pre:.2%} | Recall: {val_recall:.2%} | F1: {val_f1:.2%}\n')

        file.write(f"Epoch {epoch+1}/{num_epochs}\n")
        file.write(f"Train Loss: {train_loss_meter.avg:.4f} | Val Loss: {val_loss:.4f}\n")
        file.write(f"Classification -> Accuracy: {val_acc:.2%} | Precision: {val_pre:.2%} | Recall: {val_recall:.2%} | F1: {val_f1:.2%}\n")
        file.write("=" * 50 + "\n")

    final_acc, final_pre, final_recall, final_f1 = test(save_path, seg_model, cls_model, test_loader, fold, file)
    return (train_loss_meter.avg, val_loss, final_acc, final_pre, final_recall, final_f1)

def validate(seg_model, cls_model, val_loader, file):
    cls_model.eval()
    loss_meter = AvgMeter()
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss().to(device)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images1, images2, labels, table_info = batch['T0_image'], batch['T1_image'], batch['label'], batch['table_info']
            images1 = images1.unsqueeze(1).to(device)
            images2 = images2.unsqueeze(1).to(device)  
            labels = labels.long().to(device)
            table_info = table_info.to(device)
            
            # 特征提取
                # 要求 UNETR 的 forward 支持 return_clsfeat 参数
            y0, x, clsfeat1 = seg_model(images1, return_clsfeat=True)
            y1, x, clsfeat2 = seg_model(images2, return_clsfeat=True)
            
            # 将提取的特征和 table_info 送入新的分类模型（假设 cls_model 接收两个输入）
            class_logits = cls_model(clsfeat1,y0,clsfeat2,y1,table_info)
            loss = criterion(class_logits, labels)
            loss_meter.update(loss.item(), images1.size(0))
            
            probs = torch.softmax(class_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    file.write(f"Validation Loss: {loss_meter.avg:.4f}\n")
    file.write(f"Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}\n")
    file.write("=" * 50 + "\n")

    return (loss_meter.avg, accuracy, precision, recall, f1)

def test(model_path, seg_model, cls_model, test_loader, fold, file):
    cls_model.load_state_dict(torch.load(model_path))
    cls_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), desc="Testing"):
            images1, images2, labels, table_info = batch['T0_image'], batch['T1_image'], batch['label'], batch['table_info']
            images1 = images1.unsqueeze(1).to(device)
            images2 = images2.unsqueeze(1).to(device)  
            labels = labels.long().to(device)
            table_info = table_info.to(device)
            
            y0, x, clsfeat1 = seg_model(images1, return_clsfeat=True)
            y1, x, clsfeat2 = seg_model(images2, return_clsfeat=True)
            
            # 将提取的特征和 table_info 送入新的分类模型（假设 cls_model 接收两个输入）
            class_logits = cls_model(clsfeat1,y0,clsfeat2,y1,table_info)
            probs = torch.softmax(class_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    file.write(f"Test Results:\n")
    file.write(f"Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}\n")
    file.write("=" * 50 + "\n")

    print(f'Test Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}')
    return (accuracy, precision, recall, f1)

# 随机种子设置（保持不变）
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
    
if __name__ == '__main__':
    main()
