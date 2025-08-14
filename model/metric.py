import torch

import torch
import numpy as np
from fastdtw import fastdtw

# ==========================================================================================
# 内部帮助类，用于计算ISO_rating评价指标
# ==========================================================================================
class _ISO_ov():
    def __init__(self, line_r, line_f):
        # 确保输入是numpy array
        self.line_r = np.asarray(line_r)
        self.line_f = np.asarray(line_f)
        # 预先计算相位得分，以便后续方法复用其结果
        self.phase_score_val = self._phase_score()

    def _cora(self):
        a0=0.05
        b0=0.5
        kz=2
        curve=self.line_r
        inner_up=curve+a0*np.max(np.abs(curve))
        inner_down=curve-a0*np.max(np.abs(curve))
        outter_up=curve+b0*np.max(np.abs(curve))
        outter_down = curve - b0 * np.max(np.abs(curve))

        curve=self.line_f
        mark=np.zeros(len(curve))
        for i in range(len(curve)):
            if inner_down[i]<curve[i]<inner_up[i]:
                mark[i]=1
            elif outter_down[i]<curve[i]<outter_up[i]:
                mark[i]=(np.min([outter_up[i]-curve[i],curve[i]-outter_down[i]])/(outter_up[i]-inner_up[i]))**kz
            else:
                mark[i]=0
        return np.sum(mark)/len(curve)

    def _phase_score(self):
        yupo=0.2
        kp=1
        curve=self.line_r
        line_f=self.line_f
        n=len(curve)
        
        # 避免除以零的错误
        if np.std(curve) == 0 or np.std(line_f) == 0:
            self.r_shift = self.line_r
            self.f_shift = self.line_f
            return 1.0

        shift_range = int(np.floor(yupo * n))
        if shift_range == 0:
            self.r_shift = self.line_r
            self.f_shift = self.line_f
            return 1.0

        corrs = [np.corrcoef(line_f[i:], curve[:-i-1])[0, 1] if i != n-1 else np.corrcoef(line_f[i:], curve[:-i])[0, 1] for i in range(shift_range)]
        max_pos = np.argmax(corrs)

        self.r_shift = curve[max_pos:]
        self.f_shift = line_f[:n-max_pos]

        if max_pos == 0:
            ep = 1.0
        else:
            ep = ((yupo * n - max_pos) / (yupo * n)) ** kp
        return ep

    def _magnitude_score(self):
        yupo=0.5
        km=1
        curve_shift=self.r_shift
        f_shift=self.f_shift
        dist, path = fastdtw(curve_shift, f_shift, dist=lambda x, y: abs(x - y))
        
        sum_abs_real = np.sum(np.abs(curve_shift))
        if sum_abs_real == 0: return 1.0 # 如果真实曲线全为0，认为幅值完全匹配

        yupo_m = dist / sum_abs_real
        if yupo_m >= yupo:
            em = 0.0
        else:
            em = ((yupo - yupo_m) / yupo) ** km
        return em

    def _slope_score(self):
        yupo=2
        ks=1
        curve_shift=self.r_shift
        f_shift=self.f_shift
        n=len(curve_shift)
        interval=10
        if n <= interval: return 1.0 # 序列太短无法计算斜率

        m = int(np.ceil((n - 1) / interval))
        fake_slopes = np.zeros(m)
        real_slopes = np.zeros(m)
        
        for i in range(m - 1):
            fake_slopes[i] = (f_shift[(i + 1) * interval] - f_shift[i * interval])
            real_slopes[i] = (curve_shift[(i + 1) * interval] - curve_shift[i * interval])
        
        # 避免除以零
        sum_abs_real = np.sum(np.abs(real_slopes))
        if sum_abs_real == 0: return 1.0

        yupo_s = np.sum(np.abs(fake_slopes - real_slopes)) / sum_abs_real
        if yupo_s >= yupo:
            es = 0.0
        else:
            es = ((yupo - yupo_s) / yupo) ** ks
        return es

    def rate(self):
        wcora=0.4
        wp=0.2
        wm=0.2
        ws=0.2
        score = (wcora * self._cora() + 
                 wp * self.phase_score_val + 
                 wm * self._magnitude_score() + 
                 ws * self._slope_score())
        return score

# ==========================================================================================
# 对外暴露的Metric函数
# ==========================================================================================

def root_mean_squared_error(output, target):
    """
    计算均方根误差 (RMSE)。
    计算在所有批次、所有通道和所有时间点上的总体RMSE。
    """
    with torch.no_grad():
        loss = torch.sqrt(torch.mean((output - target)**2))
    return loss.item()

def _calculate_iso_rating_for_channel(output, target, channel_idx):
    """
    内部帮助函数，用于计算指定通道的平均ISO-rating。
    """
    with torch.no_grad():
        # 将PyTorch张量转换为Numpy数组
        # output 和 target 的形状: (batch_size, 3, 200)
        pred_waves = output.cpu().numpy()
        true_waves = target.cpu().numpy()

        batch_size = pred_waves.shape[0]
        total_score = 0.0

        for i in range(batch_size):
            # 提取指定通道的单条波形
            pred_wave = pred_waves[i, channel_idx, :]
            true_wave = true_waves[i, channel_idx, :]
            
            # 实例化并计算得分
            iso_calculator = _ISO_ov(true_wave, pred_wave)
            total_score += iso_calculator.rate()

        return total_score / batch_size if batch_size > 0 else 0.0

def iso_rating_x(output, target):
    """
    计算 X 方向波形的平均 ISO-rating。
    """
    return _calculate_iso_rating_for_channel(output, target, channel_idx=0)

def iso_rating_y(output, target):
    """
    计算 Y 方向波形的平均 ISO-rating。
    """
    return _calculate_iso_rating_for_channel(output, target, channel_idx=1)

def iso_rating_z(output, target):
    """
    计算 Z 方向波形的平均 ISO-rating。
    """
    return _calculate_iso_rating_for_channel(output, target, channel_idx=2)