import pandas as pd
import warnings
from scipy import signal
from scipy.fftpack import fft, ifft
from numpy import array, sign, zeros
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy import interpolate
from scipy import fftpack
import matplotlib as mpl
import scipy.interpolate as spi
import math
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")


def CORAL(Xs, Xt):
    '''
    Perform CORAL on the source domain features
    :param Xs: ns * n_feature, source feature
    :param Xt: nt * n_feature, target feature
    :return: New source domain features
    '''
    cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
    cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
    A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                     scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
    Xs_new = np.real(np.dot(Xs, A_coral))
    return Xs_new


# 线性插值函数
def interp_resampling(data, time, freq, start_time=0.01, down_time=0.7):
    """

    :param data: data
    :param time: time
    :param freq: target frequency
    :param start_time: start time默认0.01(J-TEXT)
    :param down_time: down time默认0.7(J-TEXT)
    :return: new data and time
    -------
    fixed by C.S.Shen, 2022.12.06
    """
    time_new = np.linspace(start_time, down_time, int((down_time - start_time) * freq))
    f = interpolate.interp1d(time, data, kind='linear', fill_value="extrapolate")
    data_new = f(np.array(time_new))
    return np.array(data_new), np.array(time_new)


def Fun(x, a, b):  # 定义拟合函数形式
    """
    :param x: 变量
    :param a: 斜率
    :param b: 偏置
    :return: 函数
    -------
    fixed by C.S.Shen, 2022.12.06
    """
    return a * x + b


def intergral_Mirnov(time_raw, data_raw, downtime=0.7):
    """

    :param time_raw: time
    :param data_raw: data
    :param downtime: downtime
    :return: time and data after intergral
    """
    f = len(time_raw) / (time_raw[-1] - time_raw[0]) if len(time_raw) > 1 else 0
    dt = 1 / f
    time_fit = time_raw[(time_raw > 0) & (time_raw < downtime)]
    data_fit = data_raw[(time_raw > 0) & (time_raw < downtime)]
    data_int = -np.cumsum(data_fit * dt / 0.069e-4)
    para, pcov = curve_fit(Fun, time_fit, data_int)
    y_fitted = data_int - para[0] * time_fit - para[1]
    return time_fit, y_fitted


def self_filter(data, sampling_freq, low_pass, high_pass):
    """
    -------
    滤波函数
    data为时间序列参数
    sampling_freq为采样频率
    low_pass, high_pass分别为高通与低通频率
    -------
    fixed by C.S.Shen, 2020.12.18
    """
    ba1 = signal.butter(8, (2 * low_pass / sampling_freq), "lowpass")
    filter_data1 = signal.filtfilt(ba1[0], ba1[1], data)
    ba2 = signal.butter(8, (2 * high_pass / sampling_freq), "highpass")
    filter_data = signal.filtfilt(ba2[0], ba2[1], filter_data1)
    return filter_data


def cal_avg_fre(data, time, chip_time=5e-3, overlap=0, low_fre=500, high_fre=5e4, step_fre=500, max_number=3, mean_th=5e-3, var_th=1e-13):
    """
    -------
    用于计算平均频率，幅值与相位
    data为Mirnov探针信号
    time为时间轴
    chip_time为fft的窗口时间长度，默认为5ms
    overlap为切割时间窗时的重叠率，默认为0
    low_fre为所需最低频率，默认为500Hz
    high_fre为所需最高频率，默认为50kHz
    step_fre为选取最大频率时的步长，默认为500Hz
    max_number为选取最大频率的个数，默认为3个
    mean_th为幅值平均值的阈值，低于该阈值则认为不存在模式
    var_th为频率间的方差阈值，默认为1e-13
    -------
    C.S.Shen, 2022.12.06
    """
    fs = 200 / (time[199] - time[0])  # Sampling frequency
    len_window = int(chip_time * fs)  # window length
    f_low = int(low_fre * chip_time)  # lowest frequency
    f_high = int(high_fre * chip_time)  # Highest frequency
    f_step = int(step_fre * chip_time)  # select max_frequency length
    number_max = int(max_number)  # maximum fre number
    slice_time, slice_data = time_slice(time, data, noverlap=overlap, slice_length=len_window)
    avg_fre = []
    avg_amp = []
    avg_pha = []
    for i in range(len(slice_time)):
        amp, fre, pha = scs_fft(slice_data[i], fs)
        amp_chosen = amp[f_low:f_high]
        f_chosen = fre[f_low:f_high]
        var_amp = np.var(amp_chosen)
        index_ch = amp_chosen.argsort()[::-1][0:number_max]
        mean_amp = np.mean(amp_chosen[index_ch])
        # 判断是否有明显模式
        TM_fre = np.zeros(max_number)
        if var_amp < var_th or mean_amp < mean_th:
            frequency, amplitude, phase = 0, 0, 0
        else:
            Piece = f_step
            index_piece = []
            max_piece = []
            for i in range(len(amp_chosen) // Piece):
                csd_piece = np.abs(amp_chosen[i * Piece:(i + 1) * Piece])
                max_piece.append(max(20 * np.log(csd_piece)))
                index_piece.append(csd_piece.argsort()[::-1][0] + f_low + i * Piece)
            tmp = np.array(max_piece)
            index_max = tmp.argsort()[::-1][0:number_max]
            N_index_max = len(index_max)
            index = np.zeros(N_index_max, dtype=np.int)
            for ii in range(N_index_max):
                index[ii] = index_piece[index_max[ii]]
            frequency, amplitude, phase = [], [], []
            for ind in index:
                frequency.append(fre[ind])
                amplitude.append(amp[ind])
                phase.append(pha[ind])
        avg_fre_slice, avg_pha_slice, avg_amp_slice = 0, 0, 0
        for iii in range(len(frequency)):
            avg_fre_slice = avg_fre_slice + frequency[iii] * amplitude[iii] / np.sum(amplitude)
            avg_amp_slice = np.mean(amplitude)
            avg_pha_slice = avg_pha_slice + phase[iii] * amplitude[iii] / np.sum(amplitude)
        avg_fre.append(avg_fre_slice)
        avg_amp.append(avg_amp_slice)
        avg_pha.append(avg_pha_slice)
    return slice_time, avg_fre, avg_amp, avg_pha


def cal_mode_number(data1, data2, time, chip_time=5e-3, overlap=0, down_number=8, low_fre=2e3, high_fre=1e5, step_fre=3e3, max_number=3, var_th=1e-13, real_angle=15, coherence_th=0.95):
    """
    -------
    用于计算模数
    data1，data2为两道Mirnov探针信号（尽可能近）
    time为他们的时间轴
    chip_time为做互功率谱时的窗口时间长度，默认为5ms
    overlap为切割时间窗时的重叠率，默认为0
    down_number为做互功率谱时的降点数（down_number = 1，即取了全时间窗的点，即FFT窗仅为一个），默认为8
    low_fre为所需最低频率，默认为2kHz
    high_fre为所需最高频率，默认为100kHz
    step_fre为选取最大频率时的步长，默认为3kHz
    max_number为选取最大频率的个数，默认为3个
    var_th为频率间的方差阈值，默认为1e-13
    real_angle为两道Mirnov探针间的极向空间角度差，默认为15°
    coherence_th为互相关系数阈值，默认为0.95
    -------
    C.S.Shen, 2020.12.18
    """
    # set parameters
    dt = time[1] - time[0]  # time interval
    fs = 200 / (time[199] - time[0])  # Sampling frequency
    len_window = int(chip_time * fs)  # window length
    length = len(time)  # length of the window time
    Chip = length // len_window  # number of the chip in the window
    f_low = int(low_fre * chip_time)  # lowest frequency
    f_high = int(high_fre * chip_time)  # Highest frequency
    f_step = int(step_fre * chip_time)  # select max_frequency length
    number_max = int(max_number)  # maximum fre number
    m = []  # m value
    fre = []  # frequency
    # filter
    data1 = self_filter(data1, fs, high_fre, low_fre)
    data2 = self_filter(data2, fs, high_fre, low_fre)
    tmp_var = []
    # slice
    slice_time, chip_data1 = time_slice(time, data1, noverlap=overlap, slice_length=len_window)
    slice_time, chip_data2 = time_slice(time, data2, noverlap=overlap, slice_length=len_window)
    for i in range(len(slice_time)):
        # calculate cross spectral density
        (f, csd) = signal.csd(chip_data1[i], chip_data2[i], fs=fs, window='hann', nperseg=len_window // down_number, scaling='density')
        (f_coherence, coherence) = signal.coherence(chip_data1[i], chip_data2[i], fs=fs, window='hann', nperseg=len_window // down_number)
        abs_csd = np.abs(csd)
        phase_csd = np.angle(csd) * 180 / np.pi
        angle_csd = np.where(coherence > coherence_th, phase_csd, 0)
        csd_chosen = csd[f_low // down_number:f_high // down_number]
        f_chosen = f[f_low // down_number:f_high // down_number]
        var_csd = np.var(np.abs(csd_chosen))
        tmp_var.append(var_csd)
        ch = np.abs(csd_chosen / np.max(np.abs(csd_chosen)))  # 求出归一化之后的互功率谱均值，若最大频率大于最小频率，且小于ch_th就证明存在明显的模式, 目前使用方差与互相关系数做约束
        index_ch = ch.argsort()[::-1][0:number_max]
        # 判断是否有明显模式
        TM_fre = np.zeros(max_number)
        if var_csd < var_th:
            m_t = 0
            TM_fre[0] = 0
        else:
            Piece = f_step // down_number
            index_piece = []
            max_piece = []
            for i in range(len(csd_chosen) // Piece):
                csd_piece = np.abs(csd_chosen[i * Piece:(i + 1) * Piece])
                max_piece.append(max(20 * np.log(csd_piece)))
                index_piece.append(csd_piece.argsort()[::-1][0] + f_low // down_number + i * Piece)
            tmp = np.array(max_piece)
            index_max = tmp.argsort()[::-1][0:number_max]
            N_index_max = len(index_max)
            index = np.zeros(N_index_max, dtype=np.int)
            for ii in range(N_index_max):
                index[ii] = index_piece[index_max[ii]]
            TM_fre = []
            TM_amp = []
            TM_phi = []
            for ind in index:
                TM_fre.append(f[ind])
                TM_amp.append(abs_csd[ind])
                TM_phi.append(angle_csd[ind])
            m_tmp = np.zeros(len(TM_phi), dtype=np.float)
            m_t = 0
            for iii in range(len(TM_phi)):
                if TM_phi[iii] < 0:
                    TM_phi[iii] = TM_phi[iii] + 360
                else:
                    if TM_phi[iii] > 360:
                        TM_phi[iii] = TM_phi[iii] - 360
                m_tmp[iii] = TM_phi[iii] / real_angle * TM_amp[iii] / np.sum(TM_amp)
                m_t = m_t + m_tmp[iii]
        m.append(m_t)
        fre.append(TM_fre[0])
    return slice_time, m, fre


def deg2deg(deg0, deg):
    """
    -------
    change degree from 0 - 360 to -180 - 180
    -------
    C.S.Shen, 2020.10.17
    """
    deg1 = deg0 * 0
    mm = np.size(deg0)
    for mm1 in range(1, mm):
        deg1[mm1] = deg0[mm1] - math.floor((deg0[mm1] + (360 - deg)) / 360) * 360
    return deg1


def n_1_mode(theta, br, deg):
    """
    -------
    用于计算n=1模式幅值与相位（认为不存在高n分量）
    输入为相对角度180°两个锁模探针的位置与数据，deg为其相对角度180°
    输出为求得的n=1幅值与相位
    -------
    br = amp*cos(theta+phase)
    C.S.Shen, 2020.10.17
    """
    theta1 = theta[0] / 180 * math.pi
    theta2 = theta[1] / 180 * math.pi
    D = math.sin(theta1 - theta2)
    br1 = br[0]
    br2 = br[1]
    amp = (br1 ** 2 + br2 ** 2 - 2 * br1 * br2 * math.cos(theta1 - theta2)) ** 0.5 / abs(math.sin(theta1 - theta2))
    cos_phi = (-br2 * math.cos(theta1) + br1 * math.cos(theta2)) / D
    sin_phi = (br2 * math.sin(theta1) - br1 * math.sin(theta2)) / D
    tanPhi = sin_phi / cos_phi
    # phase of origin is -(phs + 2 * pi * f * t)
    # phase of b ^ max is pi / 2 - (phs + 2 * pi * f * t)
    # the variable in sine function
    dlt0 = np.zeros(len(tanPhi), dtype=np.float)
    for i in range(len(tanPhi)):
        dlt0[i] = math.atan(tanPhi[i]) / math.pi * 180 + 180 * np.floor((1 - np.sign(cos_phi[i])) / 2) - 90
    # the variable in cosine function, so it is also the phase of b_theta maximum.
    phase = deg2deg(-dlt0, deg)
    # the phase of b ^ max
    return amp, phase


def locked_mode(shot, time, vbr0, theta=None):
    """
    -------
    用于计算时间序列的n=1模式幅值与相位
    shot为该炮炮号，影响NS值
    time为时间轴
    vbr0为4 * len(time) 的数组，为4个锁模探针的时间序列
    theta为2组对减后的环向空间角度
    -------
    C.S.Shen, 2020.12.18
    """
    if theta is None:
        theta = [67.5, 157.5]
    tau_br = [10e-3, 10e-3, 10e-3, 10e-3]
    br_Saddle = np.zeros((4, len(time)), dtype=np.float)
    for j1 in range(len(tau_br)):
        br_Saddle[j1] = vbr0[j1] / tau_br[j1] * 1e4
    br_odd = np.zeros((2, len(time)), dtype=np.float)  # 创建2维数组存放诊断数据
    amp = np.zeros(len(time), dtype=np.float)
    phase = np.zeros(len(time), dtype=np.float)
    br_odd[0] = br_Saddle[0] - br_Saddle[2]
    br_odd[1] = br_Saddle[1] - br_Saddle[3]
    amp, phase = n_1_mode(theta, br_odd, 180)
    return amp, phase


# 中位点采样的信号进行加工处理
def mean_down_sampling(data, time, freq):
    """
    :param data: data
    :param time: time
    :param freq: 目标采样率
    :return: 降采样后的时间与数据
    """
    fs = len(data) / (time[-1] - time[0]) if len(time) > 1 else 0
    fs_int = round(fs / 1000) * 1000
    len_chip = fs_int / freq
    processed_data = []
    processed_time = []
    section = math.ceil(len(data) / len_chip)
    for j in range(section):
        if j < (section - 1):
            mean_data = np.mean(data[int(len_chip * j):int(len_chip * j + len_chip)])
            middle_time = time[int(len_chip * j + len_chip / 2)]
            processed_data.append(mean_data)
            processed_time.append(middle_time)
        else:
            Templen = len(data[int(len_chip * j):])
            mean_data = np.mean(data[int(len_chip * j):])
            middle_time = time[int(len_chip * j + Templen / 2)]
            processed_data.append(mean_data)
            processed_time.append(middle_time)
    return np.array(processed_data), np.array(processed_time)


# 中位点采样的信号进行加工处理
def middle_down_sampling(data, time, freq):
    fs = len(data) / (time[-1] - time[0]) if len(time) > 1 else 0
    chiplen = fs / freq
    processed_data = []
    processed_time = []
    section = math.ceil(len(data) / chiplen)
    for j in range(section):
        if j < (section - 1):
            middle_data = data[int(chiplen * j + chiplen / 2)]
            middle_time = time[int(chiplen * j + chiplen / 2)]
            processed_data.append(middle_data)
            processed_time.append(middle_time)
        else:
            Templen = len(data[int(chiplen * j):])
            middle_data = data[int(chiplen * j + Templen / 2)]
            middle_time = time[int(chiplen * j + Templen / 2)]
            processed_data.append(middle_data)
            processed_time.append(middle_time)
    return np.array(processed_data), np.array(processed_time)


def factors_profile(data, time, n_channel, fs):
    """
    -------
    用于计算存在剖面信号的空间偏度、峰度与方差
    data为n_channel * len(time)维度的时间序列
    n_channel为该诊断的通道数
    fs为预期采样率，要低于真实采样率
    如果fs为-1则证明无需降采样
    -------
    偏度大于0向高场侧偏移
    峰度大于3代表比正态分布更高
    C.S.Shen, 2020.12.18
    """
    if fs < 0:
        skew = np.zeros(len(time), dtype=np.float)
        kurt = np.zeros(len(time), dtype=np.float)
        var = np.zeros(len(time), dtype=np.float)
        mean = np.zeros(len(time), dtype=np.float)
        data = np.array(data)
        for j in range(len(time)):
            skew[j] = pd.Series(data[:, j]).skew()
            kurt[j] = pd.Series(data[:, j]).kurt()
            var[j] = np.var(data[:, j])
            mean[j] = np.mean(data[:, j])
        time_down = time
    else:
        time_down = []
        data_down = np.zeros((int(n_channel), int(round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
        for i in range(n_channel):
            data_down[i], time_down = mean_down_sampling(data[i], time, fs)
        skew = np.zeros(len(time_down), dtype=np.float)
        kurt = np.zeros(len(time_down), dtype=np.float)
        var = np.zeros(len(time_down), dtype=np.float)
        mean = np.zeros(len(time_down), dtype=np.float)
        for j in range(len(time_down)):
            skew[j] = pd.Series(data_down[:, j]).skew()
            kurt[j] = pd.Series(data_down[:, j]).kurt()
            var[j] = np.var(data_down[:, j])
            mean[j] = np.mean(data_down[:, j])
    return skew, kurt, var, mean, time_down


def scs_fft(data, fs):
    fft_data = fft(data)
    amp_fft = abs(fft_data) / len(data) * 2
    label_data = np.linspace(0, int(len(data) / 2) - 1, int(len(data) / 2))
    amp = amp_fft[0:int(len(data) / 2)]
    amp[0] = 0
    fre = label_data / len(data) * fs
    phase = np.angle(fft_data)[0:int(len(data) / 2)]
    return amp, fre, phase


def time_slice(time, data, noverlap=0.5, slice_length=10):
    if noverlap == 1:
        slice_time = time
        slice_data = data
    else:
        data_length = len(data)
        slice_data = []
        slice_time = []
        nstep = (1 - noverlap) * slice_length
        Nwindows = math.floor((data_length / slice_length - noverlap) / (1 - noverlap))
        for k in range(Nwindows):
            i = k + 1
            start_index = int(data_length - 1 - (Nwindows - i) * nstep - (slice_length - 1))
            end_index = int(data_length - 1 - (Nwindows - i) * nstep)
            unit = data[start_index:end_index]
            unit_time = time[int(data_length - 1 - (Nwindows - i) * nstep)]
            slice_data.append(unit)
            slice_time.append(unit_time)
    return slice_time, slice_data


def it2bt(data):
    """
    -------
    EAST装置用于计算Bt
    data为it
    -------
    C.S.Shen, 2021.11.03
    """
    Bt = (4 * math.pi * 1e-7) * data * (16 * 130) / (2 * math.pi * 1.8)
    return Bt


def envelope(data):
    index = list(range(len(data)))
    # 获取极值点
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])

    # 将极值点拟合为曲线
    ipo3_max = spi.splrep(max_peaks, data[max_peaks], k=3)  # 样本点导入，生成参数
    iy3_max = spi.splev(index, ipo3_max)  # 根据观测点和样条参数，生成插值

    ipo3_min = spi.splrep(min_peaks, data[min_peaks], k=3)  # 样本点导入，生成参数
    iy3_min = spi.splev(index, ipo3_min)  # 根据观测点和样条参数，生成插值
    '''
    f_max = interpolate.interp1d(max_peaks, data[max_peaks], kind='linear')
    iy3_max = f_max(time)
    f_min = interpolate.interp1d(min_peaks, data[min_peaks], kind='linear')
    iy3_min = f_min(time)
    '''
    # 计算平均包络线
    iy3_mean = (iy3_max + iy3_min) / 2
    return iy3_max, iy3_min, iy3_mean


def subtract_drift(data, time):
    index = np.where(time < 0)
    mean = np.mean(data[index])
    data = data - mean
    return data


def smooth2nd(data, M):  # x 为一维数组
    K = round(M / 2 - 0.1)  # M应为奇数，如果是偶数，则取大1的奇数
    lenX = len(data)
    data_smooth = np.zeros(lenX)
    if lenX < 2 * K + 1:
        print('数据长度小于平滑点数')
        data_smooth = data
    else:
        for NN in range(0, lenX, 1):
            startInd = max([0, NN - K])
            endInd = min(NN + K + 1, lenX)
            data_smooth[NN] = np.mean(data[startInd:endInd])
    return data_smooth


def cal_p_rad(data_1, data_2, time, n_channel, fs):
    if fs < 0:
        data_down_1 = np.array(data_1)
        data_down_2 = np.array(data_2)
        time_down = time
    else:
        time_down = []
        data_down_1 = np.zeros((int(n_channel), int(round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
        data_down_2 = np.zeros((int(n_channel), int(round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
        for i in range(n_channel):
            data_down_1[i], time_down = middle_down_sampling(data_1[i], time, fs)
            data_down_2[i], time_down = middle_down_sampling(data_2[i], time, fs)
    KVT = np.array([0.72, 0.7597, 0.7730, 0.7611, 0.7921, 0.8373, 0.8911, 0.9666, 1.0000, 0.9409, 0.9851, 0.9805, 0.9443, 0.9264, 0.9084, 0.7])
    KVD = np.array([0.66, 0.6823, 0.6870, 0.6703, 0.6716, 0.7158, 0.7593, 0.9000, 0.8686, 0.7979, 0.8607, 0.8864, 0.9073, 0.7865, 0.8732, 0.85])
    rvt = np.array([-18, 2.5, 22, 42, 64, 85, 105, 127, 147.5, 168, 189, 208.5, 227.5, 246.5, 264.5, 282]) * -1
    rvd = np.array([-19, 1.5, 23, 43, 66, 89, 110.5, 130, 151, 171.5, 193, 212, 232, 251, 268, 286])
    smo = 5  # smo表征对信号的平均点数，AXUV采样率为50K，smo越大，则一些高频现象会看不到；但是太小，信号又不是很好，所以根据需求定义
    R = 1.05  # 大半径单位m
    Aap = 0.0008 * 0.005  # 狭缝面积，单位m ^ 2
    Adet = 0.002 * 0.005  # 探测器面积，单位m ^ 2
    dxf = 0.054  # 探测器到狭缝距离，单位m
    PAi = np.zeros((int(n_channel), int(round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
    PFi = np.zeros((int(n_channel), int(round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
    ya = np.zeros((int(n_channel), int(round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
    yb = np.zeros((int(n_channel), int(round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
    for j in range(n_channel):
        PAi[j] = smooth2nd(data_down_1[j] / 0.26, smo)
        PFi[j] = smooth2nd(data_down_2[j] / 0.26, smo)
        if j <= n_channel // 2:
            ya[j] = PAi[j] / KVT[j] / 2e5  # 200000是放大器倍数，前八道是2 * 10 ^ 5
            yb[j] = PFi[j] / KVD[j] / 2e5  # 200000是放大器倍数，前八道是2 * 10 ^ 5
        else:
            ya[j] = PAi[j] / KVT[j] / 5e5  # 500000是放大器倍数，后八道是5 * 10 ^ 5
            yb[j] = PFi[j] / KVD[j] / 5e5  # 500000是放大器倍数，后八道是5 * 10 ^ 5
    ya[0] = ya[1] * 0.98
    yb[1] = 0.5 * (yb[0] + yb[2])
    index = np.where(time_down < 0)
    for j in index:
        k = ya[1][j] / yb[1][j]
        for i in range(n_channel):
            ya[i][j] = yb[i][j] * k  # 两阵列校准
    sumv1, sumv2, sumv3, sumv4, sumv5 = 0, 0, 0, 0, 0
    for j in range(n_channel - 1):
        i = j + 1
        sumv1 = (rvt[i] * ya[i]) + (rvd[i] * yb[i]) + sumv1
        if i == 1:
            sumv2 = sumv2  # 高场侧
            sumv3 = np.abs(rvt[i] - rvd[i]) * ya[i] + sumv3  # 低场侧
            sumv5 = np.abs(rvt[i] - rvd[i]) * ya[i] * rvt[i] + sumv5  # 一个位置权重
        else:
            sumv2 = np.abs(rvt[i] - rvt[i - 1]) * ya[i] + sumv2
            sumv3 = np.abs(rvd[i] - rvd[i - 1]) * yb[i] + sumv3
            sumv5 = np.abs(rvt[i] - rvt[i - 1]) * ya[i] * rvt[i] + abs(rvd[i] - rvd[i - 1]) * yb[i] * rvd[i] + sumv5
        sumv4 = sumv4 + ya[i] + yb[i]
    sumvt = (sumv2 + sumv3) / 1e6
    P_rad = sumvt * (4 * np.pi * dxf ** 2) / (Adet * Aap) * (2 * np.pi * R) / 2.2
    return P_rad, time_down


def sum_ne(time, data, n_channel, fs):
    if fs < 0:
        data_down = np.array(data)
        time_down = time
    else:
        time_down = []
        data_down = np.zeros((int(n_channel), int(np.round((fs * (time[-1] - time[0])) / 100) * 100)), dtype=np.float)
        for i in range(n_channel):
            data_down[i], time_down = middle_down_sampling(data[i], time, fs)
    A = np.zeros(len(time_down), dtype=np.float)
    for i in range(len(time_down)):
        A[i] = np.sum(data_down[:, i])
    return A, time_down


def cal_p_total(ip_raw, vl_raw, time_ip, time_vl):
    ip, time = interp_resampling(ip_raw, time_ip, 10e3)
    vl, time = interp_resampling(vl_raw, time_vl, 10e3)
    P_tot = np.multiply(ip, vl)
    return P_tot, time


def find_period(data):
    mean_data = np.mean(data)
    peak_indices_max, _ = find_peaks(data, distance=5)
    cont_max = 0
    for ind in peak_indices_max:
        if data[ind] < mean_data:
            cont_max = cont_max + 1
        else:
            cont_max = cont_max

    peak_indices_min, _ = find_peaks(-data, distance=5)
    cont_min = 0
    for ind in peak_indices_min:
        if data[ind] > mean_data:
            cont_min = cont_min + 1
        else:
            cont_min = cont_min
    mode_number = max(len(peak_indices_max) - cont_max, len(peak_indices_min) - cont_min)
    return mode_number


def SVD_avg_fre_mode_number(data, time, chip_time=5e-3, overlap=0, high_fre=3e4, max_number=2, var_th=1e-14):
    """
        -------
        SVD分解后的平均频率与模式
        data为一个阵列的Mirnov探针信号[time_dim, channel_dim]
        time为时间轴
        chip_time为fft的窗口时间长度，默认为5ms
        overlap为切割时间窗时的重叠率，默认为0
        low_fre为所需最低频率，默认为500Hz
        high_fre为所需最高频率，默认为50kHz
        step_fre为选取最大频率时的步长，默认为500Hz
        max_number为选取最大频率的个数，默认为3个
        var_th为频率间的方差阈值，默认为1e-13
        -------
        C.S.Shen, 2022.12.06
        """
    fs = len(time) / (time[-1] - time[0])  # Sampling frequency
    len_window = int(chip_time * fs)  # window length
    number_max = int(max_number)
    slice_time, slice_data = time_slice(time, data, noverlap=overlap, slice_length=len_window)
    avg_fre = []
    avg_amp = []
    avg_mn = []
    for i in range(len(slice_time)):
        u, s, v = np.linalg.svd(slice_data[i], full_matrices=0)
        frequency_all = []
        amplitude_all = []
        mode_number_all = []
        for j in range(number_max):
            amp, fre, pha = scs_fft(u[:, 2 * j], fs)
            var_amp = np.var(amp)
            peak_indices, _ = find_peaks(amp, distance=5)
            amp_peaks = amp[peak_indices]
            index_peaked = amp_peaks.argsort()[::-1]
            fre_peak = fre[peak_indices[index_peaked]][0]
            amp_peak = amp[peak_indices[index_peaked]][0]
            mean_amp_peak = np.mean(amp_peaks)
            var_amp_peak = np.var(amp_peaks)
            # 判断是否有明显模式
            # if var_amp < var_th or fre_peak > high_fre or amp_peak / 8 < mean_amp_peak:
            if fre_peak > high_fre or (amp_peak / 10) < mean_amp_peak or (amp[peak_indices[index_peaked]][-1] * 2) > mean_amp_peak:
                frequency, amplitude, mode_number = 0, 0, 0
            else:
                frequency = fre_peak
                amplitude = math.sqrt(s[2 * j] ** 2 + s[2 * j + 1] ** 2)
                mode_number = find_period(v[2 * j, :])
            frequency_all.append(frequency)
            amplitude_all.append(amplitude)
            mode_number_all.append(mode_number)
        avg_fre_slice, avg_amp_slice, avg_mn_slice = 0, 0, 0
        for k in range(len(frequency_all)):
            avg_fre_slice = avg_fre_slice + frequency_all[k] * amplitude_all[k] / np.sum(amplitude_all)
            avg_mn_slice = avg_mn_slice + mode_number_all[k] * amplitude_all[k] / np.sum(amplitude_all)
        avg_amp_slice = np.mean(amplitude_all)
        avg_fre.append(avg_fre_slice)
        avg_amp.append(avg_amp_slice)
        avg_mn.append(avg_mn_slice)
    avg_mn = np.nan_to_num(np.array(avg_mn))
    avg_fre = np.nan_to_num(np.array(avg_fre))
    avg_amp = np.nan_to_num(np.array(avg_amp))
    return np.array(slice_time), avg_fre, avg_amp, avg_mn
