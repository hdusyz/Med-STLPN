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


class LungNoduleDataset(Dataset):
    def __init__(self, csv_data, data_dir,seg_dir,text_data,normalize=True,transform=transform1,augment_minority_class=True):
        self.data_dir = data_dir
        self.csv_data = csv_data
        self.text_data = dict(zip(text_data['pid'], text_data[['race', 'cigsmok', 'gender', 'age']].values.tolist()))
        self.normalize = normalize
        self.seg_dir = seg_dir
        self.transform = transform  # 增强变换
        self.augment_minority_class = augment_minority_class
        # 确保使用了重置索引后的数据
        self.csv_data.reset_index(drop=True, inplace=True)

        #self.subject_ids = self.csv_data['Subject ID']
        self.subject_ids = self.csv_data['Subject ID'].unique()
    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        if idx >= len(self.subject_ids):
            raise IndexError(f"Index {idx} is out of bounds for axis 0 with size {len(self.subject_ids)}")
        #(f"Total records in csv_data: {len(self.csv_data)}")
        #(f"Unique subject IDs: {len(self.subject_ids)}")

        subject_id = self.subject_ids[idx]
        subject_data = self.csv_data[self.csv_data['Subject ID'] == subject_id]
        
        T0_row = subject_data[subject_data['study_yr'] == 'T0']
        T1_row = subject_data[subject_data['study_yr'] == 'T1']
        T2_row = subject_data[subject_data['study_yr'] == 'T2']
        
        

        if T0_row.empty or T1_row.empty or T2_row.empty:
            raise ValueError(f"Missing data for subject {subject_id}")
        
        T0_file = f"{subject_id}_T0.npy"
        T1_file = f"{subject_id}_T1.npy"
        T2_label = T2_row.iloc[0]['label']
        T2_label=int(T2_label)
        T0_seg_file = f"{subject_id}_T0_seg.npy"
        T1_seg_file = f"{subject_id}_T1_seg.npy"

        T0_path = os.path.join(self.data_dir, T0_file)
        T1_path = os.path.join(self.data_dir, T1_file)
        
        T0_path_seg = os.path.join(self.seg_dir, T0_seg_file)
        T1_path_seg = os.path.join(self.seg_dir, T1_seg_file)
        
        T0_image = np.load(T0_path).astype(np.float32)
        T1_image = np.load(T1_path).astype(np.float32)
        
        T0_seg = np.load(T0_path_seg).astype(np.float32)
        T1_seg = np.load(T1_path_seg).astype(np.float32)
        
        if self.normalize:
            T0_image = self.normalize_image(T0_image)
            T1_image = self.normalize_image(T1_image)

        # 如果是少数类并且需要进行增强
        if self.augment_minority_class and T2_label == 1 and self.transform:
            T0_image = self.transform(T0_image)
            T1_image = self.transform(T1_image)

        T0_image = torch.tensor(np.load(T0_path), dtype=torch.float32,requires_grad=True)
        T1_image = torch.tensor(np.load(T1_path), dtype=torch.float32,requires_grad=True)
        
        T0_seg = torch.tensor(np.load(T0_path_seg), dtype=torch.float32,requires_grad=True)
        T1_seg = torch.tensor(np.load(T1_path_seg), dtype=torch.float32,requires_grad=True)
        
        label = torch.tensor(T2_label, dtype=torch.float32,requires_grad=True)
        # 获取与当前subject_id对应的文本数据
        text_input = self.text_data.get(subject_id)
        if text_input is None:
            raise ValueError(f"No text data found for Subject ID: {subject_id}")
        
        text_input = torch.tensor(text_input, dtype=torch.long)
        return T0_image, T1_image, T0_seg, T1_seg, label
    
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
    
    
class LungNoduleDataset1(Dataset):
    def __init__(self, csv_data, data_dir, normalize=True,transform=None,augment_minority_class=True):
        self.data_dir = data_dir
        self.csv_data = csv_data
        #self.text_data = dict(zip(text_data['pid'], text_data[['race', 'cigsmok', 'gender', 'age']].values.tolist()))
        self.normalize = normalize
        self.transform = transform  # 增强变换
        self.augment_minority_class = augment_minority_class
        # 确保使用了重置索引后的数据
        self.csv_data.reset_index(drop=True, inplace=True)

        #self.subject_ids = self.csv_data['Subject ID']
        self.subject_ids = self.csv_data['Subject ID'].unique()
    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        if idx >= len(self.subject_ids):
            raise IndexError(f"Index {idx} is out of bounds for axis 0 with size {len(self.subject_ids)}")
        #(f"Total records in csv_data: {len(self.csv_data)}")
        #(f"Unique subject IDs: {len(self.subject_ids)}")

        subject_id = self.subject_ids[idx]
        subject_data = self.csv_data[self.csv_data['Subject ID'] == subject_id]
        
        T0_row = subject_data[subject_data['study_yr'] == 'T0']
        T1_row = subject_data[subject_data['study_yr'] == 'T1']
        #T2_row = subject_data[subject_data['study_yr'] == 'T2']

        if T0_row.empty or T1_row.empty:
            raise ValueError(f"Missing data for subject {subject_id}")
        
        T0_file = f"{subject_id}_T0.npy"
        T1_file = f"{subject_id}_T1.npy"
        T1_label = T1_row.iloc[0]['label']
        T1_label=int(T1_label)

        T0_path = os.path.join(self.data_dir, T0_file)
        T1_path = os.path.join(self.data_dir, T1_file)

        T0_image = np.load(T0_path).astype(np.float32)
        T1_image = np.load(T1_path).astype(np.float32)
        
        if self.normalize:
            T0_image = self.normalize_image(T0_image)
            T1_image = self.normalize_image(T1_image)

        # 如果是少数类并且需要进行增强
        if self.augment_minority_class and T1_label == 1 and self.transform:
            T0_image = self.transform(T0_image)
            T1_image = self.transform(T1_image)

        T0_image = torch.tensor(np.load(T0_path), dtype=torch.float32,requires_grad=True)
        T1_image = torch.tensor(np.load(T1_path), dtype=torch.float32,requires_grad=True)
        
        label = torch.tensor(T1_label, dtype=torch.float32,requires_grad=True)
        # 获取与当前subject_id对应的文本数据
        #text_input = self.text_data.get(subject_id)
        #if text_input is None:
            #raise ValueError(f"No text data found for Subject ID: {subject_id}")
        
        #text_input = torch.tensor(text_input, dtype=torch.long)
        return T0_image, T1_image, label 
    
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
    
    
class segDataset(Dataset):
    def __init__(self, csv_data, data_dir, seg_dir, text_data, normalize=True, transform=transform1, augment_minority_class=True):
        self.data_dir = data_dir
        self.csv_data = csv_data
        specific_colunm = ['pid', 'race', 'cigsmok', 'gender', 'age', 'scr_res0', 'scr_iso0']
        self.normalize = normalize
        self.seg_dir = seg_dir
        self.transform = transform  # 增强变换
        self.augment_minority_class = augment_minority_class
        # 确保使用了重置索引后的数据
        self.csv_data.reset_index(drop=True, inplace=True)
        # deal with the table data
        self.text_data = text_data[specific_colunm].fillna('NA').astype('category')
        self.num_cat = []
        for col in specific_colunm:
            if col != 'pid':
                self.text_data[col] = LabelEncoder().fit_transform(self.text_data[col])
                self.num_cat.append(len(self.text_data[col].unique()))
        self.specific_colunm = specific_colunm[1:]

        #self.subject_ids = self.csv_data['Subject ID']
        self.subject_ids = self.csv_data['Subject ID'].unique()
    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
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
        #T2_row = subject_data[subject_data['study_yr'] == 'T2']
        
        

        if T0_row.empty or T1_row.empty :
            raise ValueError(f"Missing data for subject {subject_id}")
        
        T0_file = f"{subject_id}_T0.npy"
        T1_file = f"{subject_id}_T1.npy"
        T1_label = T1_row.iloc[0]['label']
        T1_label=int(T1_label)
        T0_seg_file = f"{subject_id}_T0.npy"
        T1_seg_file = f"{subject_id}_T1.npy"

        T0_path = os.path.join(self.data_dir, T0_file)
        T1_path = os.path.join(self.data_dir, T1_file)
        
        T0_path_seg = os.path.join(self.seg_dir, T0_seg_file)
        T1_path_seg = os.path.join(self.seg_dir, T1_seg_file)
        
        T0_image = np.load(T0_path).astype(np.float32)
        T1_image = np.load(T1_path).astype(np.float32)
        
        T0_seg = np.load(T0_path_seg).astype(np.float32)
        T1_seg = np.load(T1_path_seg).astype(np.float32)
        T0_seg = np.transpose(T0_seg, (2, 0, 1))  # 将轴从 (512, 512, 16) 转为 (16, 512, 512)
        T1_seg = np.transpose(T1_seg, (2, 0, 1))  # 将轴从 (512, 512, 16) 转为 (16, 512, 512)
        
        if self.normalize:
            T0_image = self.normalize_image(T0_image)
            T1_image = self.normalize_image(T1_image)

        # 如果是少数类并且需要进行增强
        if self.augment_minority_class and T1_label == 1 and self.transform:
            T0_image = self.transform(T0_image)
            T1_image = self.transform(T1_image)

        T0_image = torch.tensor(np.load(T0_path), dtype=torch.float32,requires_grad=True)
        T1_image = torch.tensor(np.load(T1_path), dtype=torch.float32,requires_grad=True)
        
        #T0_seg = torch.tensor(np.load(T0_seg), dtype=torch.float32,requires_grad=True)
        #T1_seg = torch.tensor(np.load(T1_seg), dtype=torch.float32,requires_grad=True)
        T0_seg = torch.tensor(T0_seg, dtype=torch.float32,requires_grad=True)
        T1_seg = torch.tensor(T1_seg, dtype=torch.float32,requires_grad=True)
        
        label = torch.tensor(T1_label, dtype=torch.float32,requires_grad=True)

        return T0_image, T1_image, T0_seg, T1_seg, label, table_info
    
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
    

class tset2d(Dataset):
    def __init__(self, csv_data, data_dir, normalize=True, transform=transform1):
        self.data_dir = data_dir
        self.csv_data = csv_data
        self.normalize = normalize
        self.transform = transform
        self.csv_data.reset_index(drop=True, inplace=True)

        self.subject_ids = self.csv_data['Subject ID'].unique()

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        if idx >= len(self.subject_ids):
            raise IndexError(f"Index {idx} is out of bounds for axis 0 with size {len(self.subject_ids)}")

        subject_id = self.subject_ids[idx]
        subject_data = self.csv_data[self.csv_data['Subject ID'] == subject_id]

        T1_row = subject_data[subject_data['study_yr'] == 'T1']

        if T1_row.empty:
            raise ValueError(f"Missing data for subject {subject_id}")

        T1_label = T1_row.iloc[0]['label']
        T1_label = int(T1_label)

        # 获取T1组的3D图像
        T1_file = f"{subject_id}_T1.npy"
        T1_path = os.path.join(self.data_dir, T1_file)

        # 确保文件存在
        if not os.path.exists(T1_path):
            raise FileNotFoundError(f"File not found: {T1_path}")

        # 读取3D图像
        T1_image = np.load(T1_path).astype(np.float32)

        # 提取第8张切片（假设第8张切片的索引为7）
        if T1_image.shape[0] < 8:
            raise ValueError(f"3D image for subject {subject_id} does not have 8 slices.")
        
        T1_slice = T1_image[7]  # 提取第8张切片（索引从0开始）

        if self.normalize:
            T1_slice = self.normalize_image(T1_slice)

        if self.transform:
            T1_slice = self.transform(T1_slice)

        T1_slice = torch.tensor(T1_slice, dtype=torch.float32, requires_grad=True)
        label = torch.tensor(T1_label, dtype=torch.float32, requires_grad=True)

        return T1_slice, label

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


import os
import numpy as np
import torch
from torch.utils.data import Dataset

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



import numpy as np
import os
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

class LungNoduleDatasetchazhi(Dataset):
    def __init__(self, csv_data, data_dir, text_data, target_size=(64, 64, 64), normalize=True, transform=None, augment_minority_class=True):
        self.data_dir = data_dir
        self.csv_data = csv_data
        self.target_size = target_size  # 目标大小
        self.normalize = normalize
        self.transform = transform  # 增强变换
        self.augment_minority_class = augment_minority_class
        # 确保使用了重置索引后的数据
        self.csv_data.reset_index(drop=True, inplace=True)
        self.subject_ids = self.csv_data['Subject ID'].unique()

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        if idx >= len(self.subject_ids):
            raise IndexError(f"Index {idx} is out of bounds for axis 0 with size {len(self.subject_ids)}")

        subject_id = self.subject_ids[idx]
        subject_data = self.csv_data[self.csv_data['Subject ID'] == subject_id]
        
        T0_row = subject_data[subject_data['study_yr'] == 'T0']
        T1_row = subject_data[subject_data['study_yr'] == 'T1']

        if T0_row.empty or T1_row.empty:
            raise ValueError(f"Missing data for subject {subject_id}")
        
        T0_file = f"{subject_id}_T0.npy"
        T1_file = f"{subject_id}_T1.npy"
        T1_label = T1_row.iloc[0]['label']
        T1_label = int(T1_label)

        T0_path = os.path.join(self.data_dir, T0_file)
        T1_path = os.path.join(self.data_dir, T1_file)

        T0_image = np.load(T0_path).astype(np.float32)
        T1_image = np.load(T1_path).astype(np.float32)

        # 使用双三次插值调整图像大小
        T0_image = self.resize_image(T0_image, target_size=(16, 64, 64))  # 调整为 16x64x64
        T1_image = self.resize_image(T1_image, target_size=(16, 64, 64))  # 调整为 

        if self.normalize:
            T0_image = self.normalize_image(T0_image)
            T1_image = self.normalize_image(T1_image)

        # 如果是少数类并且需要进行增强
        if self.augment_minority_class and T1_label == 1 and self.transform:
            T0_image = self.transform(T0_image)
            T1_image = self.transform(T1_image)

        T0_image = torch.tensor(T0_image, dtype=torch.float32, requires_grad=True)
        T1_image = torch.tensor(T1_image, dtype=torch.float32, requires_grad=True)
        
        label = torch.tensor(T1_label, dtype=torch.float32, requires_grad=True)
        
        return T0_image, T1_image, label 

    def resize_image(self, image, target_size):
        """使用双三次插值方法调整图像大小"""
        # 计算缩放因子
        zoom_factors = [
            target_size[i] / image.shape[i] for i in range(3)
        ]
        resized_image = zoom(image, zoom_factors, order=3)  # order=3表示使用双三次插值
        return resized_image

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
    
class LN_text_Dataset(Dataset):
    def __init__(self, csv_data, data_dir,text_data,normalize=True,transform=None,augment_minority_class=True):
        specific_colunm = ['pid', 'race', 'cigsmok', 'gender', 'age', 'scr_res0', 'scr_iso0']
        self.data_dir = data_dir
        self.csv_data = csv_data
        self.normalize = normalize
        self.transform = transform  # 增强变换
        self.augment_minority_class = augment_minority_class
        # 确保使用了重置索引后的数据
        self.csv_data.reset_index(drop=True, inplace=True)
        # deal with the table data
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
        #T2_row = subject_data[subject_data['study_yr'] == 'T2']

        if T0_row.empty or T1_row.empty:
            raise ValueError(f"Missing data for subject {subject_id}")
        
        T0_file = f"{subject_id}_T0.npy"
        T1_file = f"{subject_id}_T1.npy"
        T1_label = T1_row.iloc[0]['label']
        T1_label=int(T1_label)

        T0_path = os.path.join(self.data_dir, T0_file)
        T1_path = os.path.join(self.data_dir, T1_file)

        T0_image = np.load(T0_path).astype(np.float32)
        T1_image = np.load(T1_path).astype(np.float32)
        
        if self.normalize:
            T0_image = self.normalize_image(T0_image)
            T1_image = self.normalize_image(T1_image)

        # # 如果是少数类并且需要进行增强
        # if self.augment_minority_class and T1_label == 1:
        #     if self.transform:
        # # 使用传入的自定义 transform
        #         T0_image = self.transform(T0_image)
        #         T1_image = self.transform(T1_image)
        #     else:
        # # 如果没有传入 transform，使用默认的增强
        #         T0_image = self.apply_transform(T0_image)
        #         T1_image = self.apply_transform(T1_image)

        T0_image = torch.tensor(T0_image, dtype=torch.float32, requires_grad=True)
        T1_image = torch.tensor(T1_image, dtype=torch.float32, requires_grad=True)
        
        label = torch.tensor(T1_label, dtype=torch.float32, requires_grad=True)
        batch = {}
        batch['T0_image'], batch['T1_image'], batch['label'], batch['table_info'] = T0_image, T1_image, label, table_info
        return batch 
    
    def normalize_image(self, image):
        """Normalize image to zero mean and unit variance."""
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        else:
            image = image - mean  # 全0图像
        return image

    def apply_transform(self, image):
        """应用简单的3D数据增强"""
        # 随机水平翻转
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()  # 水平翻转
        # 随机垂直翻转
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()  # 垂直翻转
        # 随机旋转90度
        if random.random() > 0.5:
            image = np.rot90(image, k=1, axes=(1, 2)).copy()
        # 添加随机噪声
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.01, image.shape)
            image = image + noise
        return image
    
class MultiScaleLungNoduleDataset(Dataset):
    def __init__(self, csv_data, global_dir, roi_dir, normalize=True, transform=None, augment_minority_class=True):
        """
        Multi-scale dataset class to load both global and ROI images from separate directories.

        Args:
            csv_data (DataFrame): DataFrame containing metadata for loading images and labels.
            global_dir (str): Path to the directory containing global images.
            roi_dir (str): Path to the directory containing ROI images.
            normalize (bool): Whether to normalize the images to zero mean and unit variance.
            transform (callable, optional): Transformation function for image augmentation.
            augment_minority_class (bool): Whether to augment the minority class during training.
        """
        self.global_dir = global_dir
        self.roi_dir = roi_dir
        self.csv_data = csv_data
        self.normalize = normalize
        self.transform = transform
        self.augment_minority_class = augment_minority_class

        # Ensure the CSV data has a reset index
        self.csv_data.reset_index(drop=True, inplace=True)
        self.subject_ids = self.csv_data['Subject ID'].unique()

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        if idx >= len(self.subject_ids):
            raise IndexError(f"Index {idx} is out of bounds for axis 0 with size {len(self.subject_ids)}")

        subject_id = self.subject_ids[idx]
        subject_data = self.csv_data[self.csv_data['Subject ID'] == subject_id]

        T0_row = subject_data[subject_data['study_yr'] == 'T0']
        T1_row = subject_data[subject_data['study_yr'] == 'T1']

        if T0_row.empty or T1_row.empty:
            raise ValueError(f"Missing data for subject {subject_id}")

        # File names (consistent naming in different directories)
        T0_file = f"{subject_id}_T0.npy"
        T1_file = f"{subject_id}_T1.npy"

        T1_label = int(T1_row.iloc[0]['label'])

        # File paths for global and ROI images
        T0_global_path = os.path.join(self.global_dir, T0_file)
        T1_global_path = os.path.join(self.global_dir, T1_file)
        T0_roi_path = os.path.join(self.roi_dir, T0_file)
        T1_roi_path = os.path.join(self.roi_dir, T1_file)

        # Load images
        T0_global_image = np.load(T0_global_path).astype(np.float32)
        T1_global_image = np.load(T1_global_path).astype(np.float32)
        T0_roi_image = np.load(T0_roi_path).astype(np.float32)
        T1_roi_image = np.load(T1_roi_path).astype(np.float32)

        # Normalize images
        if self.normalize:
            T0_global_image = self.normalize_image(T0_global_image)
            T1_global_image = self.normalize_image(T1_global_image)
            T0_roi_image = self.normalize_image(T0_roi_image)
            T1_roi_image = self.normalize_image(T1_roi_image)

        # Apply data augmentation for the minority class
        if self.augment_minority_class and T1_label == 1 and self.transform:
            T0_global_image = self.transform(T0_global_image)
            T1_global_image = self.transform(T1_global_image)
            T0_roi_image = self.transform(T0_roi_image)
            T1_roi_image = self.transform(T1_roi_image)

        # Convert to PyTorch tensors
        T0_global_image = torch.tensor(T0_global_image, dtype=torch.float32, requires_grad=True)
        T1_global_image = torch.tensor(T1_global_image, dtype=torch.float32, requires_grad=True)
        T0_roi_image = torch.tensor(T0_roi_image, dtype=torch.float32, requires_grad=True)
        T1_roi_image = torch.tensor(T1_roi_image, dtype=torch.float32, requires_grad=True)
        label = torch.tensor(T1_label, dtype=torch.float32, requires_grad=True)

        return T1_global_image, T1_roi_image, label

    def normalize_image(self, image):
        """
        Normalize image to zero mean and unit variance.
        """
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        else:
            image = image - mean
        return image
