# %% 对比分析函数
import numpy as np
import matplotlib.pyplot as plt

def compare_sampling_methods():
    """
    对比Sobol采样和随机采样的效果
    """
    print("\n" + "="*80)
    print("=== Sobol采样 vs 随机采样对比分析 ===")
    print("="*80)
    
    # 加载两个数据集
    try:
        sobol_data = np.load('distribution.npz')
        random_data = np.load('distribution_Random.npz')
        print("✓ 成功加载两个数据集")
    except FileNotFoundError as e:
        print(f"✗ 文件加载失败: {e}")
        return
    
    # 提取关键的三维参数进行对比
    print("\n--- 碰撞工况参数三维空间填充质量对比 ---")
    
    def extract_normalized_3d_points(data, label):
        """提取并标准化三维点"""
        velocity = data['impact_velocity']
        angle = data['impact_angle']
        btf = data['btf']
        
        velocity_norm = (velocity - 25) / (65 - 25)
        angle_norm = (angle - (-60)) / (60 - (-60))
        btf_norm = (btf - 10) / (100 - 10)
        
        points = np.column_stack([velocity_norm, angle_norm, btf_norm])
        print(f"{label}: {len(points)} 个三维点")
        return points
    
    sobol_points = extract_normalized_3d_points(sobol_data, "Sobol采样")
    random_points = extract_normalized_3d_points(random_data, "随机采样")
    
    # 简单的对比指标
    print("\n--- 基本统计对比 ---")
    
    # 1. 最近邻距离对比
    from scipy.spatial.distance import pdist, squareform
    
    def analyze_nearest_neighbor(points, label):
        distances = pdist(points, metric='euclidean')
        distance_matrix = squareform(distances)
        np.fill_diagonal(distance_matrix, np.inf)
        nn_distances = np.min(distance_matrix, axis=1)
        
        mean_nn = np.mean(nn_distances)
        std_nn = np.std(nn_distances)
        cv_nn = std_nn / mean_nn
        
        print(f"{label}:")
        print(f"  最近邻距离均值: {mean_nn:.6f}")
        print(f"  最近邻距离标准差: {std_nn:.6f}")
        print(f"  变异系数: {cv_nn:.6f}")
        return cv_nn
    
    sobol_cv = analyze_nearest_neighbor(sobol_points, "Sobol采样")
    random_cv = analyze_nearest_neighbor(random_points, "随机采样")
    
    print(f"\n变异系数对比 (越小越好): Sobol={sobol_cv:.6f} vs 随机={random_cv:.6f}")
    if sobol_cv < random_cv:
        print("✓ Sobol采样在最近邻距离均匀性方面表现更好")
    else:
        print("✗ 随机采样在最近邻距离均匀性方面表现更好")
    
    # 2. 边缘分布均匀性对比
    print("\n--- 边缘分布均匀性对比 (KS检验) ---")
    from scipy.stats import ks_2samp
    
    uniform_ref = np.random.uniform(0, 1, len(sobol_points))
    dimensions = ['velocity', 'angle', 'btf']
    
    for i, dim_name in enumerate(dimensions):
        sobol_ks = ks_2samp(sobol_points[:, i], uniform_ref)[1]
        random_ks = ks_2samp(random_points[:, i], uniform_ref)[1]
        
        print(f"{dim_name}:")
        print(f"  Sobol p值: {sobol_ks:.6f}")
        print(f"  随机 p值: {random_ks:.6f}")
        
        if sobol_ks > random_ks:
            print(f"  ✓ Sobol采样更接近均匀分布")
        else:
            print(f"  ✗ 随机采样更接近均匀分布")
    
    # 3. 网格覆盖对比
    print("\n--- 网格覆盖对比 ---")
    
    def analyze_grid_coverage(points, label, grid_size=10):
        grid_counts = np.zeros((grid_size, grid_size, grid_size))
        grid_indices = np.floor(points * grid_size).astype(int)
        grid_indices = np.clip(grid_indices, 0, grid_size - 1)
        
        for i in range(len(points)):
            x, y, z = grid_indices[i]
            grid_counts[x, y, z] += 1
        
        occupied_grids = np.sum(grid_counts > 0)
        total_grids = grid_size ** 3
        occupancy_rate = occupied_grids / total_grids
        
        print(f"{label}:")
        print(f"  网格占用率: {occupancy_rate:.1%}")
        print(f"  占用网格数: {occupied_grids}/{total_grids}")
        
        return occupancy_rate
    
    sobol_occupancy = analyze_grid_coverage(sobol_points, "Sobol采样")
    random_occupancy = analyze_grid_coverage(random_points, "随机采样")
    
    if sobol_occupancy > random_occupancy:
        print("✓ Sobol采样具有更好的网格覆盖率")
    else:
        print("✗ 随机采样具有更好的网格覆盖率")
    
    # 4. 可视化对比
    print("\n--- 生成对比可视化 ---")
    
    fig = plt.figure(figsize=(20, 10))
    
    # Sobol采样3D散点图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(sobol_points[:, 0], sobol_points[:, 1], sobol_points[:, 2], 
               alpha=0.6, s=20, c='blue')
    ax1.set_title('Sobol采样 - 三维空间分布', fontsize=16)
    ax1.set_xlabel('Velocity (normalized)')
    ax1.set_ylabel('Angle (normalized)')
    ax1.set_zlabel('BTF (normalized)')
    
    # 随机采样3D散点图
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2], 
               alpha=0.6, s=20, c='red')
    ax2.set_title('随机采样 - 三维空间分布', fontsize=16)
    ax2.set_xlabel('Velocity (normalized)')
    ax2.set_ylabel('Angle (normalized)')
    ax2.set_zlabel('BTF (normalized)')
    
    plt.tight_layout()
    plt.show()
    
    # 生成2D投影对比
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig2.suptitle('二维投影对比 - 上行：Sobol采样，下行：随机采样', fontsize=16)
    
    # 三个二维投影
    projections = [
        (0, 1, 'Velocity vs Angle'),
        (0, 2, 'Velocity vs BTF'),
        (1, 2, 'Angle vs BTF')
    ]
    
    for i, (dim1, dim2, title) in enumerate(projections):
        # Sobol采样
        axes[0, i].scatter(sobol_points[:, dim1], sobol_points[:, dim2], 
                          alpha=0.6, s=15, c='blue')
        axes[0, i].set_title(f'Sobol - {title}')
        axes[0, i].grid(True, alpha=0.3)
        
        # 随机采样
        axes[1, i].scatter(random_points[:, dim1], random_points[:, dim2], 
                          alpha=0.6, s=15, c='red')
        axes[1, i].set_title(f'Random - {title}')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== 对比总结 ===")
    print("理论上，Sobol采样应该在以下方面表现更好：")
    print("1. 更均匀的空间分布（更低的最近邻距离变异系数）")
    print("2. 更好的维度覆盖（更高的网格占用率）")
    print("3. 更稳定的边缘分布")
    print("4. 更好的低差异性质")
    
    return {
        'sobol_cv': sobol_cv,
        'random_cv': random_cv,
        'sobol_occupancy': sobol_occupancy,
        'random_occupancy': random_occupancy
    }

# 运行对比分析
if __name__ == '__main__':
    comparison_results = compare_sampling_methods()