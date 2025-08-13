import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def process_and_save_pulses(waveform_dir, case_id_list, output_path, downsample_indices=None):
    """
    处理、降采样并打包指定案例的碰撞波形数据。

    该函数会读取给定 case_id 列表对应的 x, y, z 三个方向的原始波形CSV文件，
    进行降采样，然后将所有数据打包保存到一个 .npz 文件中，以便于高效读取。

    :param waveform_dir: 存放原始波形CSV文件的目录。
    :param case_id_list: 需要处理的案例ID列表。
    :param output_path: 打包后的 .npz 文件保存路径。
    :param downsample_indices: 用于降采样的索引数组。如果为None，则默认从20001个点中抽取200个点。
    """
    if downsample_indices is None:
        downsample_indices = np.arange(100, 20001, 100)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed_data = {}

    print(f"开始处理 {len(case_id_list)} 个案例的波形数据...")
    for case_id in tqdm(case_id_list, desc="Processing Crash Pulses"):
        try:
            x_path = os.path.join(waveform_dir, f'x{case_id}.csv')
            y_path = os.path.join(waveform_dir, f'y{case_id}.csv')
            z_path = os.path.join(waveform_dir, f'z{case_id}.csv')

            # 确认三个方向的波形文件都存在
            if not all(os.path.exists(p) for p in [x_path, y_path, z_path]):
                print(f"警告：案例 {case_id} 的波形文件不完整，已跳过。")
                continue

            ax_full = pd.read_csv(x_path, sep='\t', header=None, usecols=[1]).values
            ay_full = pd.read_csv(y_path, sep='\t', header=None, usecols=[1]).values
            az_full = pd.read_csv(z_path, sep='\t', header=None, usecols=[1]).values

            # 使用预定义的采样索引进行降采样
            ax_sampled = ax_full[downsample_indices]
            ay_sampled = ay_full[downsample_indices]
            az_sampled = az_full[downsample_indices]

            # 将三个轴的波形数据堆叠
            waveforms_np = np.stack([ax_sampled, ay_sampled, az_sampled]).squeeze() # 形状 (3, 200)

            # 以 case_id 为键，存储处理后的波形数据
            processed_data[str(case_id)] = waveforms_np

        except FileNotFoundError:
            print(f"警告：读取案例 {case_id} 文件时出错，已跳过。")
            continue

    # 保存到 .npz 文件
    np.savez(output_path, **processed_data)
    print(f"数据处理完成，已保存至 {output_path}")

if __name__ == '__main__':
    # --- 使用示例 ---
    # 定义你的数据目录和案例ID
    DATA_DIR = 'data/' # 根据你的实际路径修改
    WAVEFORM_SUBDIR = 'L-C-V-P-1800' # 根据你的实际路径修改
    
    # 假设你已经有了完整的 case_id 列表
    # 在实际使用中，你可能需要先从目录中获取所有可用的 case_id
    # from utils.util import get_complete_case_ids
    # all_case_ids = get_complete_case_ids(os.path.join(DATA_DIR, WAVEFORM_SUBDIR))
    all_case_ids = np.arange(1, 1801)


    # 调用函数进行处理和保存
    process_and_save_pulses(
        waveform_dir=os.path.join(DATA_DIR, WAVEFORM_SUBDIR),
        case_id_list=all_case_ids,
        output_path=os.path.join(DATA_DIR, 'processed_pulses.npz')
    )