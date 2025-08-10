import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from base import BaseDataLoader # 继承自项目基类
from sklearn.preprocessing import MinMaxScaler

# ==========================================================================================
# 自定义的 Scaler 类
# ==========================================================================================
class CustomAbsMaxScaler:
    """
    一个自定义的Scaler，执行 x / max(abs(x)) 的归一化。
    它的接口模仿了sklearn的scaler，以便于替换。
    """
    def __init__(self):
        self.data_abs_max_ = None

    def fit(self, data):
        """
        计算并存储数据中的绝对值最大值。
        :param data: 一个Numpy数组。
        """
        self.data_abs_max_ = np.max(np.abs(data))
        return self

    def transform(self, data):
        """
        使用存储的绝对值最大值对数据进行归一化。
        :param data: 一个Numpy数组。
        """
        if self.data_abs_max_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call a.fit(data) first.")
        if self.data_abs_max_ == 0:
            return data # 避免除以零
        return data / self.data_abs_max_

    def inverse_transform(self, data):
        """
        进行反归一化操作。
        :param data: 一个归一化后的Numpy数组。
        """
        if self.data_abs_max_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call a.fit(data) first.")
        return data * self.data_abs_max_

#==========================================================================================
# 定制的 Dataset 类来处理碰撞数据
#==========================================================================================
class CollisionDataset(Dataset):
    """
    用于加载碰撞波形数据的自定义数据集。
    【版本说明】: 此版本将波形从20001维降采样至200维，通过在1ms, 2ms...200ms时刻进行采样。
    """
    def __init__(self, npz_path, waveform_dir, case_ids, target_scaler=None):
        """
        :param npz_path: 包含所有参数的 npz 文件路径
        :param waveform_dir: 波形数据所在目录
        :param case_ids: 要加载的案例ID列表
        :param target_scaler: 可选的目标缩放器，用于对波形进行缩放
        """
        self.waveform_dir = waveform_dir
        self.case_ids = np.array(case_ids)
        self.target_scaler = target_scaler
        all_params = np.load(npz_path)
        case_indices = self.case_ids - 1 # 将 case_ids 转换为0基索引
        v_min, v_max = 25, 65
        raw_velocities = all_params['impact_velocity'][case_indices]
        norm_velocities = (raw_velocities - v_min) / (v_max - v_min) # 归一化到[0, 1]
        a_min, a_max = -60, 60
        raw_angles = all_params['impact_angle'][case_indices]
        norm_angles = ((raw_angles - a_min) / (a_max - a_min)) - 0.5 # 归一化到[-0.5, 0.5]
        o_min, o_max = -1, 1
        raw_overlaps = all_params['overlap'][case_indices]
        norm_overlaps = ((raw_overlaps - o_min) / (o_max - o_min)) - 0.5 # 归一化到[-0.5, 0.5]
        self.features = torch.tensor(
            np.stack([norm_velocities, norm_angles, norm_overlaps], axis=1),
            dtype=torch.float32
        )
        self.sampling_indices = np.arange(100, 20001, 100) # 采样索引数组，从20001个点中抽取200个点

    def __len__(self):
        """
        返回数据集中样本的总数
        """
        return len(self.case_ids)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取一个降采样后的样本
        """
        input_features = self.features[idx]
        case_id = self.case_ids[idx]
        try:
            # 读取对应案例的波形数据
            x_path = os.path.join(self.waveform_dir, f'x{case_id}.csv')
            y_path = os.path.join(self.waveform_dir, f'y{case_id}.csv')
            z_path = os.path.join(self.waveform_dir, f'z{case_id}.csv')
            ax_full = pd.read_csv(x_path, sep='\t', header=None, usecols=[1]).values
            ay_full = pd.read_csv(y_path, sep='\t', header=None, usecols=[1]).values
            az_full = pd.read_csv(z_path, sep='\t', header=None, usecols=[1]).values
            # 使用预定义的采样索引进行降采样
            ax_sampled = ax_full[self.sampling_indices]
            ay_sampled = ay_full[self.sampling_indices]
            az_sampled = az_full[self.sampling_indices]
            waveforms_np = np.stack([ax_sampled, ay_sampled, az_sampled]).squeeze() # 三个轴的波形数据, (3, 200)
            if self.target_scaler is not None: # 如果提供了目标缩放器，则对波形进行缩放
                original_shape = waveforms_np.shape
                waveforms_reshaped = waveforms_np.reshape(-1, 1)
                waveforms_scaled = self.target_scaler.transform(waveforms_reshaped)
                waveforms_np = waveforms_scaled.reshape(original_shape)
            target_waveforms = torch.tensor(waveforms_np, dtype=torch.float32)
        except FileNotFoundError as e:
            return torch.empty_like(input_features), torch.empty(3, 200, dtype=torch.float32)
        return input_features, target_waveforms # 返回特征和波形数据

#==========================================================================================
#  DataLoader 类
#==========================================================================================
class CollisionDataLoader(BaseDataLoader):
    """
    用于加载碰撞波形数据的 DataLoader 类。
    """
    def __init__(self, data_dir, waveform_subdir, batch_size, case_ids=None, normalization_mode='none', shuffle=True, validation_split=0.1, num_workers=1, training=True):
        """
        :param data_dir: 数据目录
        :param waveform_subdir: 波形数据子目录
        :param batch_size: 批量大小
        :param case_ids: 可选的案例ID列表，如果为None则加载所有案例
        :param normalization_mode: 归一化模式，'none', 'minmax', 'absmax'之一
        :param shuffle: 是否打乱数据
        :param validation_split: 验证集比例
        :param num_workers: 数据加载的工作线程数
        :param training: 是否为训练模式
        """
        # --- 路径和 case_ids 的准备 ---
        npz_path = os.path.join(data_dir, '仿真采样', 'distribution.npz')
        waveform_dir = os.path.join(data_dir, waveform_subdir)
        if case_ids is None: self.case_ids = np.arange(1, 1801)
        else: self.case_ids = case_ids
        
        target_scaler = None
        if normalization_mode != 'none':
            # --- 加载一次数据，然后根据模式选择scaler ---
            print(f"正在为目标波形加载数据以进行归一化 (模式: {normalization_mode})...")
            all_waveforms_data = []
            sampling_indices = np.arange(100, 20001, 100)
            for case_id in self.case_ids:
                try:
                    # 为更精确计算全局值，这里将x,y,z三轴数据全部加载
                    x_path = os.path.join(waveform_dir, f'x{case_id}.csv')
                    y_path = os.path.join(waveform_dir, f'y{case_id}.csv')
                    z_path = os.path.join(waveform_dir, f'z{case_id}.csv')
                    all_waveforms_data.append(pd.read_csv(x_path, sep='\t', header=None, usecols=[1]).values[sampling_indices])
                    all_waveforms_data.append(pd.read_csv(y_path, sep='\t', header=None, usecols=[1]).values[sampling_indices])
                    all_waveforms_data.append(pd.read_csv(z_path, sep='\t', header=None, usecols=[1]).values[sampling_indices])
                except FileNotFoundError:
                    continue
            if not all_waveforms_data: raise ValueError("未能加载任何波形数据。")
            
            full_dataset_np = np.concatenate(all_waveforms_data)

            if normalization_mode == 'minmax':
                global_min = full_dataset_np.min()
                global_max = full_dataset_np.max()
                print(f"全局Min: {global_min:.4f}, 全局Max: {global_max:.4f}")
                target_scaler = MinMaxScaler(feature_range=(-1, 1))
                target_scaler.fit([[global_min], [global_max]])
            elif normalization_mode == 'absmax':
                target_scaler = CustomAbsMaxScaler()
                target_scaler.fit(full_dataset_np)
                print(f"全局绝对值Max: {target_scaler.data_abs_max_:.4f}")
            else:
                raise ValueError(f"未知的 normalization_mode: {normalization_mode}")
            
            print("Scaler拟合完毕。")

        # 实例化Dataset
        self.dataset = CollisionDataset(npz_path, waveform_dir, self.case_ids, target_scaler)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)