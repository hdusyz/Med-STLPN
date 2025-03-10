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
from tqdm import tqdm  # Import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 新增分类指标导入
from models.mst_net import *

from models.resganet import ResGANet101

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
    # 初始化数据集
    #train_set = multitimedataset(csv_data, data_dir, seg_dir, text_data, normalize=True)
    train_set = SegmentandclassDataset(csv_data, data_dir, seg_dir, normalize=True)
    # 假设数据集现返回 (image, mask, label)
    cv = KFold(n_splits=4, random_state=42, shuffle=True)
    fold = 1
    num_epochs = 50

    # 指标记录
    tr_loss = []
    val_loss = []
    test_hd95 = []
    test_asd = []
    test_ji = []
    test_dice = []
    test_acc = []      # 分类准确率
    test_pre = []      # 分类精确率
    test_recall = []   # 分类召回率
    test_f1 = []       # 分类F1分数

    # 打开文件用于写入训练结果
    with open("med_seg_model.txt", "w") as file:
        file.write("Training Results\n")
        file.write("=" * 50 + "\n")

        for train_idx, test_idx in cv.split(train_set):
            print(f"\nCross validation fold {fold}")

            # 创建数据加载器
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

            # 初始化模型和训练
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #model = model = SplitterNet().to(device)  # 模型需支持多任务输出：(seg_logits, cls)
            #model = Med_STLPN(train_set.num_cat).to(device)
            model = mst_net().to(device)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params}")
            train_loss_epoch, avg_val_loss, pre_ji, pre_dice, hd95, asd, val_acc, val_pre, val_recall, val_f1 = train(
                model, train_loader, test_loader, fold, num_epochs, file
            )

            # 记录指标
            tr_loss.append(train_loss_epoch)
            val_loss.append(avg_val_loss)
            test_ji.append(pre_ji)
            test_dice.append(pre_dice)
            test_hd95.append(hd95)
            test_asd.append(asd)
            test_acc.append(val_acc)
            test_pre.append(val_pre)
            test_recall.append(val_recall)
            test_f1.append(val_f1)
            
            fold += 1
            torch.cuda.empty_cache()

        # 最终输出结果
        print('\n', '#' * 10, '最终5折交叉验证结果', '#' * 10)
        print('Average Train Loss:{:.4f}'.format(np.mean(tr_loss)))
        print('Average Val Loss:{:.4f}'.format(np.mean(val_loss)))
        print('\nSegmentation Results:')
        print('Test Jaccard:{:.2%}±{:.4f}'.format(np.mean(test_ji), np.std(test_ji)))
        print('Test Dice:{:.2%}±{:.4f}'.format(np.mean(test_dice), np.std(test_dice)))
        print('HD95:{:.2f}±{:.4f}'.format(np.mean(test_hd95), np.std(test_hd95)))
        print('ASD:{:.2f}±{:.4f}'.format(np.mean(test_asd), np.std(test_asd)))
        print('\nClassification Results:')
        print('Accuracy:{:.2%}±{:.4f}'.format(np.mean(test_acc), np.std(test_acc)))
        print('Precision:{:.2%}±{:.4f}'.format(np.mean(test_pre), np.std(test_pre)))
        print('Recall:{:.2%}±{:.4f}'.format(np.mean(test_recall), np.std(test_recall)))
        print('F1 Score:{:.2%}±{:.4f}'.format(np.mean(test_f1), np.std(test_f1)))

        # 写入最终结果到文件
        file.write("\nFinal Results\n")
        file.write("=" * 50 + "\n")
        file.write(f"Average Train Loss: {np.mean(tr_loss):.4f}\n")
        file.write(f"Average Val Loss: {np.mean(val_loss):.4f}\n")
        file.write(f"Test Jaccard: {np.mean(test_ji):.2%}±{np.std(test_ji):.4f}\n")
        file.write(f"Test Dice: {np.mean(test_dice):.2%}±{np.std(test_dice):.4f}\n")
        file.write(f"HD95: {np.mean(test_hd95):.2f}±{np.std(test_hd95):.4f}\n")
        file.write(f"ASD: {np.mean(test_asd):.2f}±{np.std(test_asd):.4f}\n")
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

def train(model, train_loader, test_loader, fold, num_epochs, file):
    # 初始化训练组件
    seg_criterion = DiceLoss().cuda()
    class_criterion = nn.CrossEntropyLoss().cuda()  # 分类损失
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * 0.9 + 0.1
    )
    
    # 指标记录
    train_loss_pic = []
    val_loss_pic = []
    best_dice = 0.0
    save_path = f'./Result/seg/seg_{fold}_model.pth'

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss_meter = AvgMeter()
        
        for images, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            # 假设输入维度符合模型要求
            images = images.unsqueeze(1).cuda()   # 如原代码所示，增加通道维度
            masks = masks.unsqueeze(1).cuda()
            labels = labels.long().cuda()  # 分类标签应为 long 型
            
            optimizer.zero_grad()
            # 模型输出：(class_logits, seg_logits)
            seg_logits, class_logits = model(images)
            seg_loss = seg_criterion(seg_logits, masks)
            class_loss = class_criterion(class_logits, labels)
            # 采用加权和（例如：0.7分割，0.3分类）
            loss = 0.7 * seg_loss + 0.3 * class_loss
            loss.backward()
            optimizer.step()
            train_loss_meter.update(loss.item(), images.size(0))
        
        # 验证阶段，同时计算分割和分类指标
        val_loss, dice, ji, hd95, asd, val_acc, val_pre, val_recall, val_f1 = validate(model, test_loader, file)
        scheduler.step()
        
        # 保存最佳模型（以分割Dice为依据，也可自行调整）
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), save_path)
        
        train_loss_pic.append(train_loss_meter.avg)
        val_loss_pic.append(val_loss)
        
        # 打印每轮结果
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss_meter.avg:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Dice: {dice:.2%} | Jaccard: {ji:.2%} | HD95: {hd95:.2f} | ASD: {asd:.2f}')
        print(f'Classification -> Accuracy: {val_acc:.2%} | Precision: {val_pre:.2%} | Recall: {val_recall:.2%} | F1: {val_f1:.2%}\n')

        # 保存每轮结果到文件
        file.write(f"Epoch {epoch+1}/{num_epochs}\n")
        file.write(f"Train Loss: {train_loss_meter.avg:.4f} | Val Loss: {val_loss:.4f}\n")
        file.write(f"Dice: {dice:.2%} | Jaccard: {ji:.2%} | HD95: {hd95:.2f} | ASD: {asd:.2f}\n")
        file.write(f"Classification -> Accuracy: {val_acc:.2%} | Precision: {val_pre:.2%} | Recall: {val_recall:.2%} | F1: {val_f1:.2%}\n")
        file.write("=" * 50 + "\n")

    # 最终测试
    final_ji, final_dice, final_hd95, final_asd, final_acc, final_pre, final_recall, final_f1 = test(save_path, model, test_loader, fold, file)
    return (train_loss_meter.avg, val_loss, final_ji, final_dice, final_hd95, final_asd, final_acc, final_pre, final_recall, final_f1)

def validate(model, val_loader, file):
    model.eval()
    loss_meter = AvgMeter()
    dice_meter = AvgMeter()
    ji_meter = AvgMeter()
    hd_meter = AvgMeter()
    asd_meter = AvgMeter()
    
    # 用于分类指标计算
    all_preds = []
    all_labels = []
    
    seg_criterion = DiceLoss().cuda()
    class_criterion = nn.CrossEntropyLoss().cuda()
    
    with torch.no_grad():
        for images, masks, labels in tqdm(val_loader, desc="Validation"):
            images = images.unsqueeze(1).cuda()
            masks = masks.unsqueeze(1).cuda()
            labels = labels.long().cuda()
            
            outputs, class_logits = model(images)
            seg_loss = seg_criterion(outputs, masks)
            class_loss = class_criterion(class_logits, labels)
            loss = 0.7 * seg_loss + 0.3 * class_loss
            
            # 计算分割指标
            dc, jc, hdc, asdc = metric_seg(outputs, masks)
            
            loss_meter.update(loss.item(), images.size(0))
            dice_meter.update(dc, images.size(0))
            ji_meter.update(jc, images.size(0))
            hd_meter.update(hdc, images.size(0))
            asd_meter.update(asdc, images.size(0))
            
            # 计算分类指标
            # 对分类 logits 使用 softmax 并取预测类别
            probs = torch.softmax(class_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    
    # 计算 sklearn 分类指标
    sklearn_accuracy = accuracy_score(all_labels, all_preds)
    sklearn_precision = precision_score(all_labels, all_preds, average='weighted')
    sklearn_recall = recall_score(all_labels, all_preds, average='weighted')
    sklearn_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 将验证结果写入文件
    file.write(f"Validation Loss: {loss_meter.avg:.4f}\n")
    file.write(f"Validation Dice: {dice_meter.avg:.2%}\n")
    file.write(f"Validation Jaccard: {ji_meter.avg:.2%}\n")
    file.write(f"Validation HD95: {hd_meter.avg:.2f}\n")
    file.write(f"Validation ASD: {asd_meter.avg:.2f}\n")
    file.write(f"Classification -> Accuracy: {sklearn_accuracy:.2%}\n")
    file.write(f"Classification -> Precision: {sklearn_precision:.2%}\n")
    file.write(f"Classification -> Recall: {sklearn_recall:.2%}\n")
    file.write(f"Classification -> F1 Score: {sklearn_f1:.2%}\n")
    file.write("=" * 50 + "\n")

    return (
        loss_meter.avg,
        dice_meter.avg,
        ji_meter.avg,
        hd_meter.avg,
        asd_meter.avg,
        sklearn_accuracy,
        sklearn_precision,
        sklearn_recall,
        sklearn_f1
    )

def test(model_path, model, test_loader, fold, file):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    JI = []
    Dices = []
    HD95 = []
    ASD = []
    to_pil = transforms.ToPILImage()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, (image, mask, label) in tqdm(enumerate(test_loader), desc="Testing"):
            image = image.unsqueeze(1).cuda()
            label = label.long().cuda()
            # 模型输出：(class_logits, seg_logits)
            output, class_logits = model(image)
            
            # 分类处理
            probs = torch.softmax(class_logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            all_preds.extend(pred_class.detach().cpu().numpy())
            all_labels.extend(label.detach().cpu().numpy())
            
            # 分割处理
            pro = torch.sigmoid(output).squeeze().cpu().numpy()
            pro = (pro > 0.5).astype(np.uint8)
            target = mask.squeeze().cpu().numpy()
            
            tp = np.sum((pro == 1) & (target == 1))
            fp = np.sum((pro == 1) & (target == 0))
            fn = np.sum((pro == 0) & (target == 1))
            
            dice = 2 * tp / (2 * tp + fp + fn + 1e-5)
            ji = tp / (tp + fp + fn + 1e-5)
            hd95_val, asd_val = cmp_3(pro, target)
            
            Dices.append(dice)
            JI.append(ji)
            HD95.append(hd95_val)
            ASD.append(asd_val)
    
    # 计算分类指标
    sklearn_accuracy = accuracy_score(all_labels, all_preds)
    sklearn_precision = precision_score(all_labels, all_preds, average='weighted')
    sklearn_recall = recall_score(all_labels, all_preds, average='weighted')
    sklearn_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 将测试结果写入文件
    file.write(f"Test Jaccard: {np.mean(JI):.2%} ± {np.std(JI):.4f}\n")
    file.write(f"Test Dice: {np.mean(Dices):.2%} ± {np.std(Dices):.4f}\n")
    file.write(f"Test HD95: {np.mean(HD95):.2f} ± {np.std(HD95):.4f}\n")
    file.write(f"Test ASD: {np.mean(ASD):.2f} ± {np.std(ASD):.4f}\n")
    file.write(f"Classification -> Accuracy: {sklearn_accuracy:.2%}\n")
    file.write(f"Classification -> Precision: {sklearn_precision:.2%}\n")
    file.write(f"Classification -> Recall: {sklearn_recall:.2%}\n")
    file.write(f"Classification -> F1 Score: {sklearn_f1:.2%}\n")
    file.write("=" * 50 + "\n")

    print('Test Result:\nSegmentation:')
    print(f'Jaccard: {np.mean(JI):.2%} | Dice: {np.mean(Dices):.2%} | HD95: {np.mean(HD95):.2f} | ASD: {np.mean(ASD):.2f}')
    print('Classification:')
    print(f'Accuracy: {sklearn_accuracy:.2%} | Precision: {sklearn_precision:.2%} | Recall: {sklearn_recall:.2%} | F1: {sklearn_f1:.2%}')

    return (
        np.mean(JI), 
        np.mean(Dices),
        np.mean(HD95),
        np.mean(ASD),
        sklearn_accuracy,
        sklearn_precision,
        sklearn_recall,
        sklearn_f1
    )

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
    
main()
