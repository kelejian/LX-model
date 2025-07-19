# %%
import numpy as np
from scipy.stats import qmc

# --- 1. 参数定义 ---
# 您可以根据需要修改采样总数
num_samples = 1800

# 定义需要独立采样的参数维度 (共17个)
# 每个参数对应Sobol序列中的一个维度
param_dims = {
    'impact_velocity': 0,    # 碰撞速度
    'impact_angle': 1,       # 碰撞角度
    'overlap': 2,            # 重叠率
    'occupant_type': 3,      # 乘员体型
    'll1': 4,                # 一级限力值
    'll2': 5,                # 二级限力值
    'btf': 6,                # 预紧器点火时刻
    'pp': 7,                 # 预紧器抽入量
    'plp': 8,                # 腰部预紧抽入量
    'lla_status': 9,         # 二级限力切换状态
    'llattf_offset': 10,     # 二级限力切换时刻的随机偏移量
    'dz': 11,                # D环高度
    'aft': 12,               # 气囊点火时刻
    'aav_status': 13,        # 二级主动泄气孔状态
    'ttf_offset': 14,        # 二级泄气孔切换时刻的随机偏移量
    'sp': 15,                # 座椅前后位置
    'recline_angle': 16      # 座椅靠背角度
}

# --- 2. Sobol序列采样 ---
# 初始化17维的Sobol序列生成器, 同时固定加扰种子
sampler = qmc.Sobol(d=len(param_dims), scramble=True, seed=2025)

# 跳过前面的初始点 (通常跳过2^d个点，其中d是维度)
skip_points = 2 ** len(param_dims)  # 对于17维，跳过2^17 = 131072个点

# 先生成要跳过的点（但不使用）
sampler.random(n=skip_points)

# 生成 [0, 1) 范围内的标准样本点
samples_unit_cube = sampler.random(n=num_samples)

# --- 3. 缩放样本到实际参数范围 ---
# 创建一个字典来存储最终的参数值
results = {}

# 遍历每个样本点进行缩放
for i in range(num_samples):
    sample = samples_unit_cube[i]

    # --- 碰撞工况参数 ---
    results.setdefault('impact_velocity', []).append(sample[param_dims['impact_velocity']] * (65 - 25) + 25)
    results.setdefault('impact_angle', []).append(sample[param_dims['impact_angle']] * (60 - (-60)) + (-60))
    
    # 特殊处理重叠率
    overlap_val = sample[param_dims['overlap']] * 200 - 100 # 映射到 [-100, 100]
    # 根据备注: "如果恰好取到0, ±100%附近的值（小于1%的差别） 直接设为100%"
    if abs(overlap_val) < 1.0 or 1.0 - abs(overlap_val) < 1.0:
        overlap_val = 100.0
    results.setdefault('overlap', []).append(overlap_val/100) # 存储为[-1, 1]范围内的值

    # --- 乘员体征参数 ---
    # 映射到 [1, 2, 3]
    occupant_type = np.floor(sample[param_dims['occupant_type']] * 3) + 1
    results.setdefault('occupant_type', []).append(int(occupant_type))

    # --- 安全带系统 ---
    # 安全带二级限力值需要小于一级限力值
    # 方案1：条件采样，但ll2和ll1会在低值聚集，不过一维边缘分布扔均匀
    ll1_val = sample[param_dims['ll1']] * (7.0 - 2.0) + 2.0
    # results.setdefault('ll1', []).append(ll1_val)
    # # 安全带二级限力值需要小于一级限力值
    # ll2_upper_bound = min(4.5, ll1_val)
    # ll2_val = sample[param_dims['ll2']] * (ll2_upper_bound - 1.5) + 1.5
    # results.setdefault('ll2', []).append(ll2_val)
    # 方案2：拒绝采样，但ll1的边缘分布非均匀
    while True:
        # 1. 在边界框内独立、均匀地生成候选点
        ll1_candidate = sampler.random(1)[0, param_dims['ll1']] * (7.0 - 2.0) + 2.0
        ll2_candidate = sampler.random(1)[0, param_dims['ll2']] * (4.5 - 1.5) + 1.5

        # 2. 检查候选点是否满足约束
        if ll1_candidate > ll2_candidate:
            # 3. 如果满足，则接受该点并跳出循环
            ll1_val = ll1_candidate
            ll2_val = ll2_candidate
            break
        # 4. 如果不满足，循环将继续，生成新的候选点

    results.setdefault('ll1', []).append(ll1_val)
    results.setdefault('ll2', []).append(ll2_val)

    btf_val = sample[param_dims['btf']] * (100 - 10) + 10
    results.setdefault('btf', []).append(btf_val)
    results.setdefault('pp', []).append(sample[param_dims['pp']] * (100 - 40) + 40)
    results.setdefault('plp', []).append(sample[param_dims['plp']] * (80 - 20) + 20)
    # 映射到 [0, 1]
    results.setdefault('lla_status', []).append(int(np.floor(sample[param_dims['lla_status']] * 2)))
    # 计算LLATTF
    llattf_offset_val = sample[param_dims['llattf_offset']] * 100
    results.setdefault('llattf', []).append(btf_val + llattf_offset_val)
    # 映射到 [1, 2, 3, 4]
    results.setdefault('dz', []).append(int(np.floor(sample[param_dims['dz']] * 4) + 1))
    # 计算PTF (确定性)
    results.setdefault('ptf', []).append(btf_val + 7.0)

    # --- 气囊系统 ---
    aft_val = sample[param_dims['aft']] * (100 - 10) + 10
    results.setdefault('aft', []).append(aft_val)
    # 映射到 [0, 1]
    results.setdefault('aav_status', []).append(int(np.floor(sample[param_dims['aav_status']] * 2)))
    # 计算TTF
    ttf_offset_val = sample[param_dims['ttf_offset']] * 100
    results.setdefault('ttf', []).append(aft_val + ttf_offset_val)

    # --- 座椅参数 ---
    # 根据乘员体型决定座椅位置范围
    sp_sample = sample[param_dims['sp']]
    if occupant_type == 1: # 5% 假人
        sp_val = sp_sample * (110 - 10) + 10
    elif occupant_type == 2: # 50% 假人
        sp_val = sp_sample * (80 - (-80)) + (-80)
    else: # 95% 假人
        sp_val = sp_sample * (40 - (-110)) + (-110)
    results.setdefault('sp', []).append(sp_val)
    results.setdefault('recline_angle', []).append(sample[param_dims['recline_angle']] * (15 - (-10)) + (-10))


# 将列表转换为Numpy数组
for key in results:
    results[key] = np.array(results[key])

# --- 4. 保存为 .npz 文件 ---
output_filename = 'distribution.npz'
np.savez_compressed(output_filename, **results)

print(f"采样完成! {num_samples}个样本点已保存至 '{output_filename}', 包含{len(results)}个参数:")
for key in results:
    print(f"  - {key}")
# 打印一个样本作为示例
print("\n--- 采样结果示例 (第一个样本点) ---")
for key, value in results.items():
    print(f"{key:<20}: {value[0]:.4f}")

# %% 验证和可视化
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def verify_and_visualize(filepath='distribution.npz'):
    """
    加载、校验并可视化.npz文件中的采样数据。
    """
    try:
        data = np.load(filepath)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{filepath}'。请确保文件名正确且文件在当前目录下。")
        return

    print("--- 开始数据校验 ---")
    all_checks_passed = True

    # 1. 连续参数范围检查
    def check_continuous(param, min_val, max_val):
        is_valid = np.all((data[param] >= min_val) & (data[param] <= max_val))
        print(f"  - 检查 '{param}': {'通过' if is_valid else '失败!!!!!!!'}")
        return is_valid

    all_checks_passed &= check_continuous('impact_velocity', 25, 65)
    all_checks_passed &= check_continuous('impact_angle', -60, 60)
    all_checks_passed &= check_continuous('ll1', 2.0, 7.0)
    all_checks_passed &= check_continuous('ll2', 1.5, 4.5)
    all_checks_passed &= check_continuous('btf', 10, 100)
    all_checks_passed &= check_continuous('pp', 40, 100)
    all_checks_passed &= check_continuous('plp', 20, 80)
    all_checks_passed &= check_continuous('aft', 10, 100)
    all_checks_passed &= check_continuous('recline_angle', -10, 15)

    # 2. 离散参数取值检查
    def check_discrete(param, allowed_values):
        is_valid = np.all(np.isin(data[param], allowed_values))
        print(f"  - 检查 '{param}': {'通过' if is_valid else '失败!!!!!!!'}")
        return is_valid

    all_checks_passed &= check_discrete('occupant_type', [1, 2, 3])
    all_checks_passed &= check_discrete('lla_status', [0, 1])
    all_checks_passed &= check_discrete('aav_status', [0, 1])
    all_checks_passed &= check_discrete('dz', [1, 2, 3, 4])

    # 3. 特殊参数针对性检查
    # 3.1 检查重叠率不为0且不为-100%且在[-100, 100]范围内
    is_overlap_valid = np.all((data['overlap'] > -100) & (data['overlap'] <= 100))
    is_overlap_valid = np.all(data['overlap'] != 0) & np.all(data['overlap'] != -100)
    print(f"  - 检查 'overlap' (不为0且不为-100%且在[-100, 100]范围内): {'通过' if is_overlap_valid else '失败!!!!!!!'}")
    all_checks_passed &= is_overlap_valid

    # 3.2 检查座椅前后位置 (SP) 与乘员体型的依赖关系
    sp = data['sp']
    occupant_type = data['occupant_type']
    mask_5p = (occupant_type == 1)
    mask_50p = (occupant_type == 2)
    mask_95p = (occupant_type == 3)
    is_sp_valid = all([
        np.all((sp[mask_5p] >= 10) & (sp[mask_5p] <= 110)),
        np.all((sp[mask_50p] >= -80) & (sp[mask_50p] <= 80)),
        np.all((sp[mask_95p] >= -110) & (sp[mask_95p] <= 40))
    ])
    print(f"  - 检查 'sp' (与体型相关): {'通过' if is_sp_valid else '失败!!!!!!!'}")
    all_checks_passed &= is_sp_valid
    
    # 3.3 检查关联参数的计算关系
    # 使用 np.allclose 来比较浮点数，避免精度问题
    is_ptf_valid = np.allclose(data['ptf'], data['btf'] + 7.0)
    print(f"  - 检查 'ptf' (等于 btf + 7ms): {'通过' if is_ptf_valid else '失败!!!!!!!'}")
    all_checks_passed &= is_ptf_valid
    
    is_llattf_valid = np.all((data['llattf'] >= data['btf']) & (data['llattf'] <= data['btf'] + 100))
    print(f"  - 检查 'llattf' (在[btf, btf+100]内): {'通过' if is_llattf_valid else '失败!!!!!!!'}")
    all_checks_passed &= is_llattf_valid

    is_ttf_valid = np.all((data['ttf'] >= data['aft']) & (data['ttf'] <= data['aft'] + 100))
    print(f"  - 检查 'ttf' (在[aft, aft+100]内): {'通过' if is_ttf_valid else '失败!!!!!!!'}")
    all_checks_passed &= is_ttf_valid

    # 3.4 检查安全带二级限力值小于等于一级限力值
    is_ll2_valid = np.all(data['ll2'] <= data['ll1'])
    print(f"  - 检查 'll2' (小于等于 ll1): {'通过' if is_ll2_valid else '失败!!!!!!!'}")
    all_checks_passed &= is_ll2_valid


    print(f"\n--- 校验总结: {'所有检查均已通过！' if all_checks_passed else '存在未通过的检查项！'} ---\n")

    if not all_checks_passed:
        print("由于校验失败，将跳过可视化部分。")
        return
        
    print("--- 开始可视化 ---")
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

    # --- 可视化1: 关键参数的一维分布 ---
    # 选取几个有代表性的连续参数
    params_to_plot_1d = ['impact_velocity', 'impact_angle', 'btf', 'recline_angle']
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    fig1.suptitle('部分参数的一维分布直方图 (探索层)', fontsize=20)
    axes1 = axes1.flatten() # 将2x2的子图数组展平，方便遍历

    for i, param in enumerate(params_to_plot_1d):
        sns.histplot(data[param], kde=False, ax=axes1[i])
        axes1[i].set_title(f'{param} 的分布')
        axes1[i].set_xlabel('值')
        axes1[i].set_ylabel('频数')

    for ax in axes1:
        ax.title.set_fontsize(20)      # 标题字体大小
        ax.xaxis.label.set_fontsize(14)  # x轴标签字体大小
        ax.yaxis.label.set_fontsize(14)  # y轴标签字体大小
        ax.tick_params(labelsize=12)     # 刻度标签字体大小

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应主标题
    
    # --- 可视化2: 关键参数的二维散点图 ---
    # 选取几个重要的参数对，观察其在二维空间的覆盖情况
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    fig2.suptitle('部分参数的二维空间覆盖情况 (探索层)', fontsize=20)
    
    # 速度 vs 角度
    axes2[0, 0].scatter(data['impact_velocity'], data['impact_angle'], alpha=0.5, s=15)
    axes2[0, 0].set_title('速度 vs 角度')
    axes2[0, 0].set_xlabel('Impact Velocity (km/h)')
    axes2[0, 0].set_ylabel('Impact Angle (°)')

    # 一级限力 vs 二级限力
    axes2[0, 1].scatter(data['ll1'], data['ll2'], alpha=0.5, s=15)
    axes2[0, 1].set_title('一级限力 vs 二级限力')
    axes2[0, 1].set_xlabel('LL1 (kN)')
    axes2[0, 1].set_ylabel('LL2 (kN)')

    # 预紧器点火 vs 气囊点火
    axes2[1, 0].scatter(data['btf'], data['aft'], alpha=0.5, s=15)
    axes2[1, 0].set_title('预紧器点火 vs 气囊点火时刻')
    axes2[1, 0].set_xlabel('BTF (ms)')
    axes2[1, 0].set_ylabel('AFT (ms)')

    # 重叠率 vs 速度
    axes2[1, 1].scatter(data['overlap'], data['impact_velocity'], alpha=0.5, s=15)
    axes2[1, 1].set_title('重叠率 vs 速度')
    axes2[1, 1].set_xlabel('Overlap (%)')
    axes2[1, 1].set_ylabel('Impact Velocity (km/h)')

    # 设置所有子图的字体大小
    for ax in axes2.flatten():
        ax.tick_params(labelsize=14)
        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ---可视化3: 碰撞工况参数三维散点图 ---
    fig3 = plt.figure(figsize=(14, 12))
    ax = fig3.add_subplot(111, projection='3d')
    ax.scatter(data['impact_velocity'], data['impact_angle'], data['btf'], alpha=0.5)
    ax.set_title('碰撞工况参数三维散点图')
    ax.set_xlabel('Impact Velocity (km/h)')
    ax.set_ylabel('Impact Angle (°)')
    ax.set_zlabel('BTF (ms)')
    # 字体调大
    ax.tick_params(labelsize=14)
    ax.title.set_fontsize(18)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)   

    # # 在可视化3部分添加更多角度的视图
    # fig3 = plt.figure(figsize=(18, 12))

    # # 创建多个子图，从不同角度观察
    # ax1 = fig3.add_subplot(221, projection='3d')
    # ax1.scatter(data['impact_velocity'], data['impact_angle'], data['btf'], alpha=0.6, s=20)
    # ax1.view_init(elev=20, azim=45)
    # ax1.set_title('视角1: 默认视角', fontsize=14)

    # ax2 = fig3.add_subplot(222, projection='3d')
    # ax2.scatter(data['impact_velocity'], data['impact_angle'], data['btf'], alpha=0.6, s=20)
    # ax2.view_init(elev=60, azim=0)
    # ax2.set_title('视角2: 侧视', fontsize=14)

    # ax3 = fig3.add_subplot(223, projection='3d')
    # ax3.scatter(data['impact_velocity'], data['impact_angle'], data['btf'], alpha=0.6, s=20)
    # ax3.view_init(elev=0, azim=90)
    # ax3.set_title('视角3: 正视', fontsize=14)

    # ax4 = fig3.add_subplot(224, projection='3d')
    # ax4.scatter(data['impact_velocity'], data['impact_angle'], data['btf'], alpha=0.6, s=20)
    # ax4.view_init(elev=90, azim=0)
    # ax4.set_title('视角4: 俯视', fontsize=14)

    # for ax in [ax1, ax2, ax3, ax4]:
    #     ax.set_xlabel('Impact Velocity (km/h)')
    #     ax.set_ylabel('Impact Angle (°)')
    #     ax.set_zlabel('BTF (ms)')
    #     ax.tick_params(labelsize=12)

    plt.tight_layout()

    print("绘图完成，请查看弹出的图表窗口。")
    plt.show()


if __name__ == '__main__':
    verify_and_visualize('distribution_Random.npz')
# %%
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import qmc, kstest, chi2
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_space_filling_quality_comprehensive(data):
    """
    全面分析碰撞工况参数三维空间填充质量 - 严格的数学验证
    """
    print("=== 严格的Sobol序列空间填充质量分析 ===")
    
    # 提取三维数据
    velocity = data['impact_velocity']
    angle = data['impact_angle'] 
    btf = data['btf']
    
    # 标准化到[0,1]³单位立方体
    # 这是所有后续分析的强制要求，确保所有参数在同一尺度下比较
    velocity_norm = (velocity - 25) / (65 - 25)
    angle_norm = (angle - (-60)) / (60 - (-60))
    btf_norm = (btf - 10) / (100 - 10)
    
    points = np.column_stack([velocity_norm, angle_norm, btf_norm])
    n_points = len(points)
    
    print(f"分析维度: 3D, 样本数: {n_points}")
    print("-" * 60)

    # --- 1. 星盘差异度 (Star Discrepancy) ---
    print("\n[指标1: 星盘差异度 (Star Discrepancy)]")
    print("解读: 这是衡量点集均匀性的黄金标准。值越接近0，代表点集在空间中的分布越均匀。\n"
          "      我们将计算Sobol样本的差异度，并与一个纯随机样本对比，以凸显其优势。")
    
    sobol_discrepancy = qmc.discrepancy(points, method='CD')
    print(f"  - Sobol样本的差异度: {sobol_discrepancy:.6f}")
    
    # 创建一个同样大小的随机样本作为对比基准
    random_points = np.random.rand(n_points, 3)
    random_discrepancy = qmc.discrepancy(random_points)
    print(f"  - 对比用随机样本的差异度: {random_discrepancy:.6f}")
    
    if sobol_discrepancy < random_discrepancy / 2:
        print("  \n结论: ✅ Sobol样本的差异度显著低于随机样本，证明其空间填充质量非常高。")
    else:
        print("  \n结论: ⚠️ Sobol样本的差异度与随机样本相比优势不明显，请检查采样过程。")
    print("-" * 60)

    # --- 2. 单维度投影的 Kolmogorov-Smirnov (K-S) 检验 ---
    print("\n[指标2: 单维度 K-S 检验]")
    print("解读: 此检验用于判断单个参数的样本分布是否符合理想的均匀分布。\n"
          "      我们会看p-value。如果p-value > 0.05，我们就有信心认为该参数的采样是均匀的。")
    
    param_names = ['速度 (Velocity)', '角度 (Angle)', 'BTF']
    all_ks_passed = True
    for i, name in enumerate(param_names):
        stat, pvalue = kstest(points[:, i], 'uniform')
        print(f"  - {name} 投影的 K-S 检验: p-value = {pvalue:.4f}")
        if pvalue <= 0.05:
            all_ks_passed = False
            print(f"    警告: {name}的p-value过低，其一维分布的均匀性不佳！")

    if all_ks_passed:
        print("  \n结论: ✅ 所有参数的一维投影均通过了均匀性检验。")
    else:
        print("  \n结论: ❌ 部分参数未通过均匀性检验，采样可能存在问题。")
    print("-" * 60)

    # --- 3. 多维卡方 (Chi-Squared) 检验 ---
    print("\n[指标3: 多维卡方 (Chi-Squared) 检验]")
    print("解读: 此检验将三维空间划分为多个小方格，检查样本点是否均匀地落入每个格子中。\n"
          "      同样，如果p-value > 0.05，说明从整体密度来看，样本是均匀分布的。")
    
    # 选择合适的网格划分数k，使得每个小方格的期望点数不低于5
    k = 0
    for k_test in range(10, 2, -1):
        if n_points / (k_test**3) >= 5.0:
            k = k_test
            break
    
    if k == 0:
        print("  - 样本量过小，无法进行有效的卡方检验。跳过此项。")
    else:
        M = k**3
        expected_freq = n_points / M
        print(f"  - 空间被划分为 {k}x{k}x{k} = {M} 个小方格，每个格子期望点数: {expected_freq:.2f}")

        observed_freq, _ = np.histogramdd(points, bins=k, range=[(0, 1), (0, 1), (0, 1)])
        
        chi2_stat = np.sum((observed_freq.flatten() - expected_freq)**2 / expected_freq)
        df = M - 1
        p_value = chi2.sf(chi2_stat, df) # sf是生存函数，等价于 1 - cdf

        print(f"  - 卡方检验统计量: {chi2_stat:.2f}, p-value = {p_value:.4f}")

        if p_value > 0.05:
            print("  \n结论: ✅ 卡方检验通过，样本点的空间密度分布是均匀的。")
        else:
            print("  \n结论: ❌ 卡方检验未通过，样本点在空间中可能存在聚集或稀疏区域。")
    print("-" * 60)
    
    # --- 4. 最近邻距离分析 (Nearest-Neighbor Distance Analysis) ---
    print("\n[指标4: 最近邻距离分析 (可视化)]")
    print("解读: 均匀分布的点集，其点与点之间的距离会比较规整。\n"
          "      如果图中出现一个非常靠近0的尖峰，说明存在点聚集的情况。Sobol序列的分布通常比\n"
          "      随机序列更窄、更集中，表明其结构更规整，没有意外的“洞”或“团”。")

    # 计算Sobol样本的最近邻距离
    nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(points)
    distances, _ = nn.kneighbors(points)
    sobol_nn_distances = distances[:, 1]

    # 计算随机样本的最近邻距离用于对比
    nn_random = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(random_points)
    distances_random, _ = nn_random.kneighbors(random_points)
    random_nn_distances = distances_random[:, 1]
    
    print(f"  - Sobol样本最近邻距离: 平均值={np.mean(sobol_nn_distances):.4f}, 标准差={np.std(sobol_nn_distances):.4f}")
    print(f"  - 随机样本最近邻距离: 平均值={np.mean(random_nn_distances):.4f}, 标准差={np.std(random_nn_distances):.4f}")
    
    # 绘图
    plt.figure(figsize=(12, 7))
    sns.kdeplot(sobol_nn_distances, label=f'Sobol Sample (std={np.std(sobol_nn_distances):.3f})', fill=True)
    sns.kdeplot(random_nn_distances, label=f'Random Sample (std={np.std(random_nn_distances):.3f})', fill=True, alpha=0.7)
    plt.title('最近邻距离分布对比 (Sobol vs. 随机)', fontsize=16)
    plt.xlabel('到最近邻居的距离', fontsize=12)
    plt.ylabel('密度', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    print("\n  结论: ✅ 请观察上图。Sobol序列的分布曲线更窄，表明其点间距更一致、结构更规整。")
    print("-" * 60)

data = np.load('distribution.npz')
# 评估 Sobol 序列的空间填充质量
analyze_space_filling_quality_comprehensive(data)

 
# %% 简单随机采样
import numpy as np
from scipy.stats import qmc

# --- 简单随机采样版本 ---
def generate_random_samples(num_samples=1800, seed=2025):
    """
    使用简单随机采样生成参数样本，用于与Sobol采样对比
    """
    # 设置随机种子确保结果可重复
    np.random.seed(seed)
    
    print(f"开始生成 {num_samples} 个随机样本...")
    
    # 创建一个字典来存储最终的参数值
    results = {}
    
    # 直接生成所需数量的随机样本
    for i in range(num_samples):
        # 生成17个[0,1)范围内的随机数
        sample = np.random.uniform(0, 1, 17)
        
        # --- 碰撞工况参数 ---
        results.setdefault('impact_velocity', []).append(sample[0] * (65 - 25) + 25)
        results.setdefault('impact_angle', []).append(sample[1] * (60 - (-60)) + (-60))
        
        # 特殊处理重叠率
        overlap_val = sample[2] * 200 - 100  # 映射到 [-100, 100]
        # 根据备注: "如果恰好取到0附近的值或-100%，直接设为100%"
        if abs(overlap_val) < 1e-6 or np.isclose(overlap_val, -100.0):
            overlap_val = 100.0
        results.setdefault('overlap', []).append(overlap_val)

        # --- 乘员体征参数 ---
        # 映射到 [1, 2, 3]
        occupant_type = np.floor(sample[3] * 3) + 1
        results.setdefault('occupant_type', []).append(int(occupant_type))

        # --- 安全带系统 ---
        # 使用拒绝采样确保 ll2 < ll1
        attempts = 0
        max_attempts = 1000  # 防止无限循环
        while attempts < max_attempts:
            # 生成新的随机数用于限力值
            ll1_rand = np.random.uniform(0, 1)
            ll2_rand = np.random.uniform(0, 1)
            
            ll1_candidate = ll1_rand * (7.0 - 2.0) + 2.0
            ll2_candidate = ll2_rand * (4.5 - 1.5) + 1.5

            # 检查候选点是否满足约束
            if ll1_candidate > ll2_candidate:
                ll1_val = ll1_candidate
                ll2_val = ll2_candidate
                break
            attempts += 1
        
        if attempts >= max_attempts:
            # 如果拒绝采样失败，使用条件采样作为后备
            ll1_val = sample[4] * (7.0 - 2.0) + 2.0
            ll2_upper_bound = min(4.5, ll1_val)
            ll2_val = sample[5] * (ll2_upper_bound - 1.5) + 1.5

        results.setdefault('ll1', []).append(ll1_val)
        results.setdefault('ll2', []).append(ll2_val)

        btf_val = sample[6] * (100 - 10) + 10
        results.setdefault('btf', []).append(btf_val)
        results.setdefault('pp', []).append(sample[7] * (100 - 40) + 40)
        results.setdefault('plp', []).append(sample[8] * (80 - 20) + 20)
        # 映射到 [0, 1]
        results.setdefault('lla_status', []).append(int(np.floor(sample[9] * 2)))
        # 计算LLATTF
        llattf_offset_val = sample[10] * 100
        results.setdefault('llattf', []).append(btf_val + llattf_offset_val)
        # 映射到 [1, 2, 3, 4]
        results.setdefault('dz', []).append(int(np.floor(sample[11] * 4) + 1))
        # 计算PTF (确定性)
        results.setdefault('ptf', []).append(btf_val + 7.0)

        # --- 气囊系统 ---
        aft_val = sample[12] * (100 - 10) + 10
        results.setdefault('aft', []).append(aft_val)
        # 映射到 [0, 1]
        results.setdefault('aav_status', []).append(int(np.floor(sample[13] * 2)))
        # 计算TTF
        ttf_offset_val = sample[14] * 100
        results.setdefault('ttf', []).append(aft_val + ttf_offset_val)

        # --- 座椅参数 ---
        # 根据乘员体型决定座椅位置范围
        sp_sample = sample[15]
        if occupant_type == 1:  # 5% 假人
            sp_val = sp_sample * (110 - 10) + 10
        elif occupant_type == 2:  # 50% 假人
            sp_val = sp_sample * (80 - (-80)) + (-80)
        else:  # 95% 假人
            sp_val = sp_sample * (40 - (-110)) + (-110)
        results.setdefault('sp', []).append(sp_val)
        results.setdefault('recline_angle', []).append(sample[16] * (15 - (-10)) + (-10))

    # 将列表转换为Numpy数组
    for key in results:
        results[key] = np.array(results[key])

    return results

# 生成随机采样数据
print("=== 生成简单随机采样数据 ===")
random_results = generate_random_samples(num_samples=1800, seed=2025)

# 保存为 .npz 文件
random_output_filename = 'distribution_Random.npz'
np.savez_compressed(random_output_filename, **random_results)

print(f"随机采样完成! {len(random_results['impact_velocity'])}个样本点已保存至 '{random_output_filename}', 包含{len(random_results)}个参数:")
for key in random_results:
    print(f"  - {key}")

# 打印一个样本作为示例
print("\n--- 随机采样结果示例 (第一个样本点) ---")
for key, value in random_results.items():
    print(f"{key:<20}: {value[0]:.4f}")

