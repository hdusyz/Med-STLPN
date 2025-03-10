from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from sklearn.calibration import LabelEncoder



# 数据增强变换
class RandomFlip(object):
    """随机翻转数据"""
    def __call__(self, sample):
        if random.random() > 0.5:
            sample = np.flip(sample, axis=0).copy()  # 水平翻转
        if random.random() > 0.5:
            sample = np.flip(sample, axis=1).copy()  # 垂直翻转
        return sample

class RandomRotation(object):
    """随机旋转数据"""
    def __call__(self, sample):
        angle = random.choice([0, 90, 180, 270])
        sample = np.rot90(sample, k=angle//90, axes=(0, 1)).copy()
        return sample

class AddGaussianNoise(object):
    """添加高斯噪声"""
    def __call__(self, sample, mean=0, std=0.05):
        noise = np.random.normal(mean, std, sample.shape)
        sample = sample + noise
        sample = np.clip(sample, 0, 1)  # 保证数据在[0, 1]之间
        return sample

# 创建数据增强管道
transform1 = transforms.Compose([
    RandomFlip(),
    RandomRotation(),
    AddGaussianNoise()
])


class SegmentandclassDataset(Dataset):
    def __init__(self, csv_data, data_dir, seg_dir, normalize=True, transform=None):
        """
        自定义数据集类，用于加载图像和分割文件。

        Args:
            csv_data (DataFrame): 包含样本元数据的 DataFrame，需包含 'Subject ID' 和 'study_yr' 两列。
            data_dir (str): 存储源文件的目录路径。
            seg_dir (str): 存储分割文件的目录路径。
            normalize (bool): 是否对图像数据进行归一化处理。
            transform (callable, optional): 用于数据增强的变换函数。
        """
        self.csv_data = csv_data
        self.data_dir = data_dir
        self.seg_dir = seg_dir
        self.normalize = normalize
        self.transform = transform

        # 提取所有唯一样本的索引
        self.csv_data.reset_index(drop=True, inplace=True)
        self.subject_ids = self.csv_data['Subject ID'].unique()
    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        # 获取当前行数据
        row = self.csv_data.iloc[idx]
        subject_id = row['Subject ID']
        study_yr = row['study_yr']
        label = row['label']  # 读取标签

        # 构造文件路径
        file_name = f"{subject_id}_T1.npy"
        image_path = os.path.join(self.data_dir, file_name)
        seg_path = os.path.join(self.seg_dir, file_name)

        # 加载数据
        image = np.load(image_path, allow_pickle=True).astype(np.float32)
        seg = np.load(seg_path, allow_pickle=True).astype(np.float32)
        
        # 调整分割图像形状
        seg = np.transpose(seg, (2, 0, 1))  # 将轴从 (512, 512, 16) 转为 (16, 512, 512)
    
        # 归一化图像
        if self.normalize:
            image = self.normalize_image(image)

        # 应用数据增强（如果有）
        if self.transform:
            image = self.transform(image)

        # 转为 PyTorch 张量
        image = torch.tensor(image, dtype=torch.float32)
        seg = torch.tensor(seg, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)  # 转为 PyTorch 张量
        
        return image, seg, label

    @staticmethod
    def normalize_image(image):
        """将图像数据归一化为零均值和单位方差。"""
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return (image - mean) / std
        return image - mean

class SegmentDataset(Dataset):
    def __init__(self, csv_data, data_dir, seg_dir, normalize=True, transform=None):
        """
        自定义数据集类，用于加载图像和分割文件。

        Args:
            csv_data (DataFrame): 包含样本元数据的 DataFrame，需包含 'Subject ID' 和 'study_yr' 两列。
            data_dir (str): 存储源文件的目录路径。
            seg_dir (str): 存储分割文件的目录路径。
            normalize (bool): 是否对图像数据进行归一化处理。
            transform (callable, optional): 用于数据增强的变换函数。
        """
        self.csv_data = csv_data
        self.data_dir = data_dir
        self.seg_dir = seg_dir
        self.normalize = normalize
        self.transform = transform

        # 提取所有唯一样本的索引
        self.csv_data.reset_index(drop=True, inplace=True)
        self.subject_ids = self.csv_data['Subject ID'].unique()
    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        # 获取当前行数据
        subject_id = self.subject_ids[idx]
        
        # 根据 Subject ID 找到对应的行数据
        row = self.csv_data[self.csv_data['Subject ID'] == subject_id].iloc[0]
        study_yr = row['study_yr']
        label = row['label']  # 读取标签

        # 构造文件路径
        file_name = f"{subject_id}_T1.npy"
        image_path = os.path.join(self.data_dir, file_name)
        seg_path = os.path.join(self.seg_dir, file_name)

        # 加载数据
        image = np.load(image_path, allow_pickle=True).astype(np.float32)
        seg = np.load(seg_path, allow_pickle=True).astype(np.float32)
        
        # 调整分割图像形状
        seg = np.transpose(seg, (2, 0, 1))  # 将轴从 (512, 512, 16) 转为 (16, 512, 512)
    
        # 归一化图像
        if self.normalize:
            image = self.normalize_image(image)

        # 应用数据增强（如果有）
        if self.transform:
            image = self.transform(image)

        # 转为 PyTorch 张量
        image = torch.tensor(image, dtype=torch.float32)
        seg = torch.tensor(seg, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)  # 转为 PyTorch 张量
        
        return image, seg

    @staticmethod
    def normalize_image(image):
        """将图像数据归一化为零均值和单位方差。"""
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return (image - mean) / std
        return image - mean


class clsDataset(Dataset):
    def __init__(self, csv_data, data_dir, normalize=True, transform=None):
        """
        自定义数据集类，用于加载图像和分割文件。

        Args:
            csv_data (DataFrame): 包含样本元数据的 DataFrame，需包含 'Subject ID' 和 'study_yr' 两列。
            data_dir (str): 存储源文件的目录路径。
            seg_dir (str): 存储分割文件的目录路径。
            normalize (bool): 是否对图像数据进行归一化处理。
            transform (callable, optional): 用于数据增强的变换函数。
        """
        self.csv_data = csv_data
        self.data_dir = data_dir
        self.normalize = normalize
        self.transform = transform

        # 提取所有唯一样本的索引
        self.csv_data.reset_index(drop=True, inplace=True)
        self.subject_ids = self.csv_data['Subject ID'].unique()
    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        # 获取当前行数据
        if idx >= len(self.subject_ids):
            raise IndexError(f"Index {idx} is out of bounds for axis 0 with size {len(self.subject_ids)}")

        subject_id = self.subject_ids[idx]
        subject_data = self.csv_data[self.csv_data['Subject ID'] == subject_id]

        T1_row = subject_data[subject_data['study_yr'] == 'T1']
        T1_file = f"{subject_id}_T1.npy"
        T1_path = os.path.join(self.data_dir, T1_file)
        T1_label = T1_row.iloc[0]['label']
        T1_label=int(T1_label)
        # 加载数据
        image = np.load(T1_path, allow_pickle=True).astype(np.float32)
        #@seg = np.load(seg_path, allow_pickle=True).astype(np.float32)
        
        # 调整分割图像形状
        #seg = np.transpose(seg, (2, 0, 1))  # 将轴从 (512, 512, 16) 转为 (16, 512, 512)
    
        # 归一化图像
        if self.normalize:
            image = self.normalize_image(image)

        # 应用数据增强（如果有）
        if self.transform:
            image = self.transform(image)

        # 转为 PyTorch 张量
        image = torch.tensor(image, dtype=torch.float32)
        #seg = torch.tensor(seg, dtype=torch.float32)
        label = torch.tensor(T1_label, dtype=torch.float32, requires_grad=True)
        
        return image, label

    @staticmethod
    def normalize_image(image):
        """将图像数据归一化为零均值和单位方差。"""
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return (image - mean) / std
        return image - mean


class multitimedataset(Dataset):
    def __init__(self, csv_data, data_dir,seg_dir,text_data,normalize=False,transform=None,augment_minority_class=True):
        specific_colunm = ['pid', 'race', 'cigsmok', 'gender', 'age', 'scr_res0', 'scr_iso0']
        self.data_dir = data_dir
        self.csv_data = csv_data
        self.seg_dir = seg_dir
        self.normalize = normalize
        self.transform = transform
        self.text_data = text_data[specific_colunm].fillna('NA').astype('category')
        self.num_cat = []
        for col in specific_colunm:
            if col != 'pid':
                self.text_data[col] = LabelEncoder().fit_transform(self.text_data[col])
                self.num_cat.append(len(self.text_data[col].unique()))
        self.specific_colunm = specific_colunm[1:]

        self.subject_ids = self.csv_data['Subject ID'].unique()

    def __len__(self):
        return len(self.subject_ids)
    def __getitem__(self, idx):
            # 获取当前行数据
        if idx >= len(self.subject_ids):
            raise IndexError(f"Index {idx} is out of bounds for axis 0 with size {len(self.subject_ids)}")
        #(f"Total records in csv_data: {len(self.csv_data)}")
        #(f"Unique subject IDs: {len(self.subject_ids)}")

        subject_id = self.subject_ids[idx]
        subject_data = self.csv_data[self.csv_data['Subject ID'] == subject_id]
        table_info = self.text_data[self.text_data['pid'] == subject_id]
        table_info = torch.tensor(table_info[self.specific_colunm].values, dtype=torch.int64)
        
        T0_row = subject_data[subject_data['study_yr'] == 'T0']
        T1_row = subject_data[subject_data['study_yr'] == 'T1']

        T0_file = f"{subject_id}_T0.npy"
        T1_file = f"{subject_id}_T1.npy"
        T1_label = T1_row.iloc[0]['label']
        T1_label=int(T1_label)

        T0_image_path = os.path.join(self.data_dir, T0_file)
        T1_iamge_path = os.path.join(self.data_dir, T1_file)

        T0_seg_path = os.path.join(self.seg_dir,T0_file)
        T1_seg_path = os.path.join(self.seg_dir,T1_file)

        T0_image = np.load(T0_image_path, allow_pickle=True).astype(np.float32)
        T1_image = np.load(T1_iamge_path, allow_pickle=True).astype(np.float32)
        if self.normalize:
            T0_image = self.normalize_image(T0_image)
            T1_image = self.normalize_image(T1_image)


        T0_seg = np.load(T0_seg_path, allow_pickle=True).astype(np.float32)
        T1_seg = np.load(T1_seg_path, allow_pickle=True).astype(np.float32)
        T0_seg = np.transpose(T0_seg, (2, 0, 1))
        T1_seg = np.transpose(T1_seg, (2, 0, 1))

        T0_image = torch.tensor(T0_image, dtype=torch.float32)
        T1_image = torch.tensor(T1_image, dtype=torch.float32)
        T0_seg = torch.tensor(T0_seg, dtype=torch.float32)
        T1_seg = torch.tensor(T1_seg, dtype=torch.float32)

        label = torch.tensor(T1_label, dtype=torch.float32)
        batch = {}
        batch['T0_image'], batch['T1_image'], batch['T0_seg'], batch['T1_seg'], batch['label'], batch['table_info'] = T0_image, T1_image, T0_seg, T1_seg, label, table_info
        return batch 

    def normalize_image(self, image):
        """Normalize image to zero mean and unit variance."""
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        else:
            # If std is zero (all values in image are the same), set to zero array
            image = image - mean
        return image



