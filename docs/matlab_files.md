# MATLAB Files

## Files

- src\optimization\optimize_costas_params.m
- src\sync\costas_loop_sync.m
- src\sync\improved_costas_sync.m
- src\sync\multi_stage_costas_sync.m
- src\sync\square_law_sync.m
- test\sync\test_sync_methods.m
- main.m

---


### src\sync\costas_loop_sync.m

```matlab
function [freq_error, snr_estimate] = costas_loop_sync(signal, fs, f_carrier)
    % Costas环法实现载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率
    %   f_carrier: 载波频率
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)
    
    % Costas环参数
    damping = 0.707;   % 临界阻尼
    noise_bw = 0.02;   % 降低噪声带宽，提高稳定性
    
    % 频率限幅设置
    freq_max = 2*pi*50/fs;  % 最大允许频率偏差±50Hz
    freq_min = -freq_max;
    
    % 初始化变量
    N = length(signal);
    phase = 0;
    freq = 0;
    freq_history = zeros(1, N);
    phase_history = zeros(1, N);
    
    % 环路滤波器系数计算
    K1 = 4 * damping * noise_bw / (1 + 2 * damping * noise_bw + noise_bw^2);
    K2 = 4 * noise_bw^2 / (1 + 2 * damping * noise_bw + noise_bw^2);
    
    % Costas环处理
    I_arm = zeros(1, N);
    Q_arm = zeros(1, N);
    error = zeros(1, N);
    
    for n = 1:N
        % 生成本地载波
        I_carrier = cos(2 * pi * f_carrier * (n-1) / fs + phase);
        Q_carrier = -sin(2 * pi * f_carrier * (n-1) / fs + phase);
        
        % I/Q解调
        I_arm(n) = signal(n) * I_carrier;
        Q_arm(n) = signal(n) * Q_carrier;
        
        % 改进的相位检测器：使用符号判决来增强抗噪性能
        error(n) = sign(I_arm(n)) * Q_arm(n);
        
        % 环路滤波器
        freq = freq + K2 * error(n);
        % 频率限幅
        freq = max(min(freq, freq_max), freq_min);
        
        phase = phase + freq + K1 * error(n);
        % 相位归一化到[-pi, pi]
        phase = mod(phase + pi, 2*pi) - pi;
        
        % 记录历史
        freq_history(n) = freq;
        phase_history(n) = phase;
    end
    
    % 计算频率误差（使用稳态频率）
    steady_state_start = floor(N * 0.7);  % 使用后30%的数据
    avg_freq_radians = mean(freq_history(steady_state_start:end));
    
    % 转换为Hz
    freq_error = avg_freq_radians * fs / (2 * pi);
    
    % 改进的SNR估计
    % 使用稳态I/Q信号的功率比
    I_steady = I_arm(steady_state_start:end);
    Q_steady = Q_arm(steady_state_start:end);
    
    % 计算信号功率（I路主要包含信号）
    signal_power = mean(I_steady.^2);
    
    % 计算噪声功率（Q路主要包含噪声）
    noise_power = mean(Q_steady.^2);
    
    % 计算SNR
    snr_estimate = 10 * log10(signal_power / noise_power);
    
    % 限制SNR估计的范围，避免不合理的值
    snr_estimate = min(max(snr_estimate, 0), 40);
    
    % 调试绘图（仅在需要时取消注释）
    % figure;
    % subplot(3,1,1);
    % plot(freq_history * fs / (2*pi));
    % title('频率历史 (Hz)');
    % xlabel('样本');
    % ylabel('频率 (Hz)');
    % 
    % subplot(3,1,2);
    % plot(phase_history);
    % title('相位历史');
    % xlabel('样本');
    % ylabel('相位 (rad)');
    % 
    % subplot(3,1,3);
    % plot(error);
    % title('误差信号');
    % xlabel('样本');
    % ylabel('误差');
end 
```


### src\sync\improved_costas_sync.m

```matlab
%{
改进的Costas环同步器

问题背景：
1. 原始Costas环存在的问题：
   - 大频偏（>5Hz）时容易失锁
   - SNR估计精度低
   - 频率估计不够平滑
   - 捕获过程不够稳定

2. 改进方案：
   a) 增加预同步阶段：
      - 使用FFT进行粗频率估计
      - 提供更好的初始频率估计值
   
   b) 自适应环路参数：
      - 捕获阶段：大带宽，快速跟踪
      - 跟踪阶段：小带宽，提高精度
   
   c) 优化相位检测器：
      - 使用arctan检测器提高精度
      - 增加判决反馈减少噪声影响
   
   d) 改进SNR估计：
      - 使用功率谱密度估计
      - 分离信号和噪声成分

3. 预期效果：
   - 扩大频率捕获范围
   - 提高频率估计精度
   - 改善SNR估计准确度
   - 增强系统稳定性
%}

function [freq_error, snr_estimate, debug_info] = improved_costas_sync(signal, fs, f_carrier)
    % 改进的Costas环载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率
    %   f_carrier: 载波频率
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)
    %   debug_info: 调试信息结构体
    
    % 预同步阶段参数
    fft_size = 2^nextpow2(length(signal));
    freq_resolution = fs/fft_size;
    
    % Costas环参数（来自优化结果）
    damping = 0.5;
    initial_noise_bw = 0.1;  % 捕获阶段带宽
    final_noise_bw = 0.02;   % 跟踪阶段带宽
    freq_max = 10;           % Hz
    
    % 预同步：FFT粗频率估计
    [initial_freq_error, initial_snr] = fft_presync(signal, fs, f_carrier, fft_size);
    
    % 初始化变量
    N = length(signal);
    phase = 0;
    freq = 2*pi*initial_freq_error/fs;  % 使用FFT估计作为初始值
    freq_history = zeros(1, N);
    phase_history = zeros(1, N);
    noise_bw_history = zeros(1, N);
    
    % 初始化I/Q分支
    I_arm = zeros(1, N);
    Q_arm = zeros(1, N);
    error = zeros(1, N);
    
    % Costas环处理
    for n = 1:N
        % 自适应噪声带宽
        progress = n/N;
        current_noise_bw = adaptive_noise_bw(initial_noise_bw, final_noise_bw, progress);
        noise_bw_history(n) = current_noise_bw;
        
        % 计算环路滤波器系数
        [K1, K2] = calculate_loop_coefficients(current_noise_bw, damping);
        
        % 生成本地载波
        I_carrier = cos(2*pi*f_carrier*(n-1)/fs + phase);
        Q_carrier = -sin(2*pi*f_carrier*(n-1)/fs + phase);
        
        % I/Q解调
        I_arm(n) = signal(n) * I_carrier;
        Q_arm(n) = signal(n) * Q_carrier;
        
        % 改进的相位检测器（arctan）
        error(n) = improved_phase_detector(I_arm(n), Q_arm(n));
        
        % 环路滤波器
        freq = freq + K2 * error(n);
        % 频率限幅
        freq = limit_frequency(freq, freq_max, fs);
        
        % 相位更新
        phase = phase + freq + K1 * error(n);
        phase = mod(phase + pi, 2*pi) - pi;
        
        % 记录历史
        freq_history(n) = freq;
        phase_history(n) = phase;
    end
    
    % 使用稳态数据计算最终结果
    [freq_error, snr_estimate] = calculate_final_estimates(freq_history, I_arm, Q_arm, fs);
    
    % 收集调试信息
    debug_info = struct(...
        'freq_history', freq_history * fs/(2*pi), ...
        'phase_history', phase_history, ...
        'noise_bw_history', noise_bw_history, ...
        'error_signal', error, ...
        'initial_freq_error', initial_freq_error, ...
        'initial_snr', initial_snr);
end

function [freq_error, snr] = fft_presync(signal, fs, f_carrier, fft_size)
    % FFT预同步
    window = hanning(length(signal))';
    windowed_signal = signal .* window;
    
    % 计算FFT
    spectrum = fft(windowed_signal, fft_size);
    freq = (0:fft_size-1)*fs/fft_size;
    magnitude = abs(spectrum(1:fft_size/2));
    
    % 在载波频率附近搜索峰值
    search_range = 50;  % Hz
    carrier_bin = round(f_carrier*fft_size/fs) + 1;
    search_start = max(1, carrier_bin - round(search_range*fft_size/fs));
    search_end = min(fft_size/2, carrier_bin + round(search_range*fft_size/fs));
    
    [~, max_idx] = max(magnitude(search_start:search_end));
    peak_freq = freq(search_start + max_idx - 1);
    
    % 计算频率误差
    freq_error = peak_freq - f_carrier;
    
    % 估计SNR
    signal_power = mean(magnitude(max_idx-2:max_idx+2).^2);
    noise_power = mean(magnitude([1:max_idx-5, max_idx+5:end]).^2);
    snr = 10*log10(signal_power/noise_power);
end

function noise_bw = adaptive_noise_bw(initial_bw, final_bw, progress)
    % 自适应调整噪声带宽
    % 使用sigmoid函数实现平滑过渡
    transition_point = 0.3;  % 30%处开始转换
    transition_width = 0.2;  % 转换区域宽度
    x = (progress - transition_point)/transition_width;
    sigmoid = 1/(1 + exp(5*x));  % sigmoid函数
    noise_bw = final_bw + (initial_bw - final_bw)*sigmoid;
end

function [K1, K2] = calculate_loop_coefficients(noise_bw, damping)
    % 计算环路滤波器系数
    K1 = 4 * damping * noise_bw / (1 + 2*damping*noise_bw + noise_bw^2);
    K2 = 4 * noise_bw^2 / (1 + 2*damping*noise_bw + noise_bw^2);
end

function error = improved_phase_detector(I, Q)
    % 改进的相位检测器
    % 使用arctan检测器，并加入判决反馈
    error = atan2(Q, I);
    
    % 加入软判决
    confidence = sqrt(I^2 + Q^2);
    error = error * tanh(confidence);  % 使用tanh限制误差幅度
end

function freq = limit_frequency(freq, freq_max, fs)
    % 频率限幅
    freq_max_rad = 2*pi*freq_max/fs;
    freq = max(min(freq, freq_max_rad), -freq_max_rad);
end

function [freq_error, snr_estimate] = calculate_final_estimates(freq_history, I_arm, Q_arm, fs)
    % 计算最终估计值
    N = length(freq_history);
    steady_state_start = floor(N * 0.7);  % 使用后30%的数据
    
    % 频率误差估计
    avg_freq_radians = mean(freq_history(steady_state_start:end));
    freq_error = avg_freq_radians * fs / (2*pi);
    
    % 改进的SNR估计
    I_steady = I_arm(steady_state_start:end);
    Q_steady = Q_arm(steady_state_start:end);
    
    % 使用功率谱密度估计
    [pxx, f] = pwelch(I_steady, [], [], [], fs);
    signal_band = abs(f) <= 5;  % 假设信号带宽为5Hz
    signal_power = mean(pxx(signal_band));
    noise_power = mean(pxx(~signal_band));
    
    snr_estimate = 10*log10(signal_power/noise_power);
    snr_estimate = min(max(snr_estimate, 0), 40);  % 限制范围
end 
```


### main.m

```matlab
% 主程序文件
% 运行载波同步系统测试

% 添加所需路径
addpath('src/sync');
addpath('src/optimization');
addpath('test/sync');
addpath('utils/common');

% 清理工作空间
clear;
clc;

% 运行测试
test_sync_methods(); 
```


### src\sync\multi_stage_costas_sync.m

```matlab
%{
多级Costas环同步器 - 第二版改进

改进背景：
第一版改进后的测试结果分析：
1. 性能表现：
   - 小频偏（0.5~2 Hz）：表现优秀，误差<0.5%
   - 中频偏（5 Hz）：表现良好，误差<0.11%
   - 大频偏（10-20 Hz）：
     * 10Hz：尚可接受，误差<2%
     * 20Hz：性能显著下降，误差约59%
   - SNR估计：在正常工作范围内准确（20-22dB）

2. 存在的问题：
   - 20Hz频偏时性能不理想
   - 高频偏时SNR估计准确度下降
   - 10Hz以上频偏时精度逐渐下降
   - 预同步阶段搜索范围可能不足

本次改进方案：
1. 多级同步策略：
   - 第一级：FFT粗搜索（±100Hz范围）
   - 第二级：分段精细搜索（±25Hz步进）
   - 第三级：Costas环精确跟踪

2. 自适应参数优化：
   - 根据频偏大小动态调整环路参数
   - 使用平滑的状态转换函数
   - 增加环路稳定性保护

3. 改进的SNR估计：
   - 多窗口谱估计
   - 自适应信号带宽检测
   - 去除异常值影响

预期效果：
1. 扩大有效频偏范围到±50Hz
2. 提高大频偏时的精度（目标<5%）
3. 改善SNR估计在全频段的准确度
4. 缩短捕获时间
%}

function [freq_error, snr_estimate, debug_info] = multi_stage_costas_sync(signal, fs, f_carrier)
    % 多级Costas环同步器
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率
    %   f_carrier: 载波频率
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)
    %   debug_info: 调试信息结构体
    
    % 第一级：FFT粗搜索
    [coarse_freq_error, initial_snr] = wide_range_fft_search(signal, fs, f_carrier);
    
    % 第二级：分段精细搜索
    [refined_freq_error, refined_snr] = fine_grid_search(signal, fs, f_carrier, coarse_freq_error);
    
    % 第三级：Costas环精确跟踪
    [final_freq_error, final_snr, tracking_info] = adaptive_costas_tracking(...
        signal, fs, f_carrier, refined_freq_error);
    
    % 返回最终结果
    freq_error = final_freq_error;
    snr_estimate = final_snr;
    
    % 收集调试信息
    debug_info = struct(...
        'coarse_stage', struct('freq_error', coarse_freq_error, 'snr', initial_snr), ...
        'fine_stage', struct('freq_error', refined_freq_error, 'snr', refined_snr), ...
        'tracking_stage', tracking_info);
end

function [freq_error, snr] = wide_range_fft_search(signal, fs, f_carrier)
    % 宽范围FFT搜索
    % 使用大窗口FFT进行±100Hz范围的粗搜索
    
    % 使用多个窗口减少频谱泄漏
    window_types = {
        @hamming,
        @hanning,
        @blackman
    };
    
    fft_size = 2^nextpow2(length(signal));
    freq = (-fft_size/2:fft_size/2-1)*fs/fft_size;
    search_range = 100;  % Hz
    
    % 多窗口平均频谱
    avg_spectrum = zeros(1, fft_size);
    for i = 1:length(window_types)
        window = window_types{i}(length(signal))';
        windowed_signal = signal .* window;
        spectrum = fftshift(abs(fft(windowed_signal, fft_size)));
        avg_spectrum = avg_spectrum + spectrum/length(window_types);
    end
    
    % 在载波频率附近搜索
    carrier_bin = find(freq >= f_carrier, 1);
    search_width = round(search_range*fft_size/fs);
    search_range = carrier_bin + (-search_width:search_width);
    search_range = search_range(search_range > 0 & search_range <= fft_size);
    
    [~, max_idx] = max(avg_spectrum(search_range));
    peak_freq = freq(search_range(max_idx));
    
    % 使用抛物线插值提高频率分辨率
    if max_idx > 1 && max_idx < length(search_range)
        alpha = avg_spectrum(search_range(max_idx-1));
        beta = avg_spectrum(search_range(max_idx));
        gamma = avg_spectrum(search_range(max_idx+1));
        offset = 0.5 * (alpha - gamma)/(alpha - 2*beta + gamma);
        peak_freq = peak_freq + offset*fs/fft_size;
    end
    
    freq_error = peak_freq - f_carrier;
    
    % 改进的SNR估计
    signal_band = abs(freq - peak_freq) <= 5;
    signal_power = mean(avg_spectrum(signal_band).^2);
    noise_power = mean(avg_spectrum(~signal_band).^2);
    snr = 10*log10(signal_power/noise_power);
end

function [freq_error, snr] = fine_grid_search(signal, fs, f_carrier, coarse_error)
    % 分段精细搜索
    % 在粗搜索结果附近进行精细网格搜索
    
    % 搜索参数
    search_width = 25;  % Hz
    step_size = 0.5;    % Hz
    segment_length = floor(fs/10);  % 0.1秒segments
    num_segments = floor(length(signal)/segment_length);
    
    % 生成搜索频率网格
    search_freqs = (coarse_error-search_width):step_size:(coarse_error+search_width);
    correlation_power = zeros(size(search_freqs));
    
    % 对每个频率假设计算相关性
    for i = 1:length(search_freqs)
        f_test = f_carrier + search_freqs(i);
        total_power = 0;
        
        % 分段处理以减少计算量
        for seg = 1:num_segments
            idx = (seg-1)*segment_length + (1:segment_length);
            t = (idx-1)/fs;
            test_signal = exp(-1j*2*pi*f_test*t);
            segment_correlation = abs(sum(signal(idx).*test_signal));
            total_power = total_power + segment_correlation;
        end
        
        correlation_power(i) = total_power/num_segments;
    end
    
    % 找到最佳匹配
    [~, max_idx] = max(correlation_power);
    freq_error = search_freqs(max_idx);
    
    % 计算SNR
    peak_power = correlation_power(max_idx);
    noise_power = mean(correlation_power([1:max_idx-5, max_idx+5:end]));
    snr = 10*log10(peak_power/noise_power);
end

function [freq_error, snr_estimate, debug_info] = adaptive_costas_tracking(...
    signal, fs, f_carrier, initial_freq_error)
    % 自适应Costas环跟踪
    % 根据频偏大小动态调整环路参数
    
    % 初始化变量
    N = length(signal);
    phase = 0;
    freq = 2*pi*initial_freq_error/fs;
    
    % 历史记录
    freq_history = zeros(1, N);
    phase_history = zeros(1, N);
    error_history = zeros(1, N);
    bw_history = zeros(1, N);
    
    % 自适应参数
    freq_error_mag = abs(initial_freq_error);
    if freq_error_mag > 10
        % 大频偏模式
        initial_bw = 0.2;
        final_bw = 0.05;
        damping = 0.5;
        freq_max = 50;
    elseif freq_error_mag > 5
        % 中频偏模式
        initial_bw = 0.1;
        final_bw = 0.02;
        damping = 0.707;
        freq_max = 25;
    else
        % 小频偏模式
        initial_bw = 0.05;
        final_bw = 0.01;
        damping = 1.0;
        freq_max = 10;
    end
    
    % Costas环处理
    for n = 1:N
        % 自适应带宽控制
        progress = n/N;
        current_bw = adaptive_bandwidth(initial_bw, final_bw, progress);
        bw_history(n) = current_bw;
        
        % 计算环路系数
        [K1, K2] = calculate_loop_coefficients(current_bw, damping);
        
        % 生成本地载波
        t = (n-1)/fs;
        I_carrier = cos(2*pi*f_carrier*t + phase);
        Q_carrier = -sin(2*pi*f_carrier*t + phase);
        
        % I/Q解调
        I = signal(n) * I_carrier;
        Q = signal(n) * Q_carrier;
        
        % 改进的相位检测器
        error = improved_phase_detector(I, Q, freq/2/pi*fs);
        error_history(n) = error;
        
        % 环路滤波器
        freq = freq + K2 * error;
        freq = limit_frequency(freq, freq_max, fs);
        
        % 相位更新
        phase = phase + freq + K1 * error;
        phase = mod(phase + pi, 2*pi) - pi;
        
        % 记录历史
        freq_history(n) = freq;
        phase_history(n) = phase;
    end
    
    % 计算最终结果
    [freq_error, snr_estimate] = calculate_final_estimates(...
        freq_history, signal, fs, f_carrier);
    
    % 调试信息
    debug_info = struct(...
        'freq_history', freq_history * fs/(2*pi), ...
        'phase_history', phase_history, ...
        'error_history', error_history, ...
        'bw_history', bw_history);
end

function bw = adaptive_bandwidth(initial_bw, final_bw, progress)
    % 改进的自适应带宽控制
    % 使用平滑的状态转换函数
    
    % 三段式转换
    if progress < 0.2
        % 快速捕获阶段
        bw = initial_bw;
    elseif progress < 0.4
        % 平滑过渡阶段
        x = (progress - 0.2)/0.2;
        bw = initial_bw + (final_bw - initial_bw) * (3*x^2 - 2*x^3);
    else
        % 稳定跟踪阶段
        bw = final_bw;
    end
end

function error = improved_phase_detector(I, Q, freq_error)
    % 改进的相位检测器
    % 加入频率自适应的判决反馈
    
    % 基本相位错误
    error = atan2(Q, I);
    
    % 信号强度
    amplitude = sqrt(I^2 + Q^2);
    
    % 频率依赖的置信度
    freq_confidence = exp(-abs(freq_error)/20);  % 20Hz特征频率
    
    % 软判决
    confidence = amplitude * freq_confidence;
    error = error * tanh(confidence);
end

function [freq_error, snr] = calculate_final_estimates(freq_history, signal, fs, f_carrier)
    % 改进的最终估计计算
    % 使用多窗口方法提高准确度
    
    N = length(freq_history);
    steady_state_start = floor(N * 0.6);  % 使用后40%的数据
    
    % 频率误差估计（使用中值滤波去除异常值）
    window_size = 101;
    freq_smooth = medfilt1(freq_history(steady_state_start:end), window_size);
    freq_error = mean(freq_smooth) * fs/(2*pi);
    
    % 改进的SNR估计
    segment_length = floor(fs/5);  % 0.2秒segments
    num_segments = floor((N-steady_state_start+1)/segment_length);
    snr_estimates = zeros(1, num_segments);
    
    for i = 1:num_segments
        start_idx = steady_state_start + (i-1)*segment_length;
        segment = signal(start_idx:start_idx+segment_length-1);
        
        % 使用Welch方法估计功率谱
        [pxx, f] = pwelch(segment, [], [], [], fs);
        
        % 自适应信号带宽
        center_freq = f_carrier + freq_error;
        signal_band = abs(f - center_freq) <= max(2, abs(freq_error)/5);
        
        signal_power = mean(pxx(signal_band));
        noise_power = mean(pxx(~signal_band));
        snr_estimates(i) = 10*log10(signal_power/noise_power);
    end
    
    % 去除异常值后平均
    snr = median(snr_estimates);
    snr = min(max(snr, 0), 40);  % 限制范围
end

function [K1, K2] = calculate_loop_coefficients(noise_bw, damping)
    % 计算环路滤波器系数
    % 输入参数:
    %   noise_bw: 噪声带宽
    %   damping: 阻尼系数
    % 输出参数:
    %   K1, K2: 环路滤波器系数
    
    % 基于二阶PLL的系数计算公式
    K1 = 4 * damping * noise_bw / (1 + 2*damping*noise_bw + noise_bw^2);
    K2 = 4 * noise_bw^2 / (1 + 2*damping*noise_bw + noise_bw^2);
end

function freq = limit_frequency(freq, freq_max, fs)
    % 频率限幅函数
    % 输入参数:
    %   freq: 当前频率（弧度/采样）
    %   freq_max: 最大允许频率（Hz）
    %   fs: 采样频率（Hz）
    % 输出参数:
    %   freq: 限幅后的频率（弧度/采样）
    
    % 将Hz转换为弧度/采样
    freq_max_rad = 2*pi*freq_max/fs;
    
    % 限幅
    freq = max(min(freq, freq_max_rad), -freq_max_rad);
end 
```


### src\optimization\optimize_costas_params.m

```matlab
%{
问题描述：
Costas环参数优化器的开发背景：
1. 当前Costas环存在以下问题：
   - 小频偏（0.5~2 Hz）时性能很好
   - 大频偏（5~20 Hz）时完全失锁
   - SNR估计普遍偏低

2. 参数选择的困境：
   - noise_bw（噪声带宽）影响跟踪速度和稳定性
   - damping（阻尼系数）影响系统响应特性
   - freq_max（最大频率限制）影响捕获范围
   这些参数之间存在相互制约关系，难以手动选择最优值

3. 优化目标：
   - 提高大频偏时的捕获性能
   - 保持小频偏时的高精度
   - 改善SNR估计准确度

优化方法：
1. 使用网格搜索遍历参数空间
2. 对每组参数进行Monte Carlo测试
3. 综合评估频率误差和SNR误差
4. 选择最优参数组合
%}

function best_params = optimize_costas_params()
    % 优化Costas环参数
    % 输出:
    %   best_params: 结构体，包含最优参数
    
    % 测试参数设置
    freq_offsets = [0.5, 1, 2, 5, 10, 20];  % Hz
    snrs = [10, 20, 30];  % dB
    fs = 1000;  % Hz
    f_carrier = 100;  % Hz
    signal_length = 10;  % 秒
    monte_carlo_runs = 10;  % 每组参数测试次数
    
    % 参数搜索范围
    noise_bw_range = [0.01, 0.02, 0.05, 0.1, 0.2];
    damping_range = [0.5, 0.707, 1.0];
    freq_max_range = [10, 25, 50, 100];  % Hz
    
    % 初始化最优结果
    best_score = -inf;
    best_params = struct();
    
    % 创建结果文件
    fid = fopen('optimization_results.txt', 'w');
    fprintf(fid, 'Costas环参数优化结果\n');
    fprintf(fid, '===================\n\n');
    
    % 参数网格搜索
    for noise_bw = noise_bw_range
        for damping = damping_range
            for freq_max = freq_max_range
                % 当前参数组合的性能统计
                total_score = 0;
                freq_errors = [];
                snr_errors = [];
                
                % 对每种测试条件进行多次Monte Carlo测试
                for f_offset = freq_offsets
                    for snr = snrs
                        for run = 1:monte_carlo_runs
                            % 生成测试信号
                            t = 0:1/fs:signal_length;
                            modulated_signal = cos(2*pi*(f_carrier + f_offset)*t);
                            noisy_signal = awgn(modulated_signal, snr);
                            
                            % 使用当前参数进行同步测试
                            [freq_error, snr_est] = test_costas_params(noisy_signal, fs, f_carrier, ...
                                noise_bw, damping, freq_max);
                            
                            % 计算误差
                            freq_err_percent = abs((freq_error - f_offset)/f_offset) * 100;
                            snr_err_db = abs(snr_est - snr);
                            
                            freq_errors = [freq_errors, freq_err_percent];
                            snr_errors = [snr_errors, snr_err_db];
                        end
                    end
                end
                
                % 计算性能得分
                % 频率误差权重更大，因为这是主要优化目标
                freq_score = -mean(freq_errors);  % 负值因为误差越小越好
                snr_score = -mean(snr_errors);
                total_score = freq_score * 0.8 + snr_score * 0.2;
                
                % 记录结果
                fprintf(fid, '参数组合:\n');
                fprintf(fid, 'noise_bw: %.3f\n', noise_bw);
                fprintf(fid, 'damping: %.3f\n', damping);
                fprintf(fid, 'freq_max: %.1f Hz\n', freq_max);
                fprintf(fid, '平均频率误差: %.2f%%\n', -freq_score);
                fprintf(fid, '平均SNR误差: %.2f dB\n', -snr_score);
                fprintf(fid, '总得分: %.2f\n\n', total_score);
                
                % 更新最优参数
                if total_score > best_score
                    best_score = total_score;
                    best_params.noise_bw = noise_bw;
                    best_params.damping = damping;
                    best_params.freq_max = freq_max;
                    best_params.score = total_score;
                end
            end
        end
    end
    
    % 记录最优参数
    fprintf(fid, '\n最优参数组合:\n');
    fprintf(fid, 'noise_bw: %.3f\n', best_params.noise_bw);
    fprintf(fid, 'damping: %.3f\n', best_params.damping);
    fprintf(fid, 'freq_max: %.1f Hz\n', best_params.freq_max);
    fprintf(fid, '最终得分: %.2f\n', best_params.score);
    
    fclose(fid);
    fprintf('优化完成，结果已保存到 optimization_results.txt\n');
end

function [freq_error, snr_estimate] = test_costas_params(signal, fs, f_carrier, noise_bw, damping, freq_max)
    % 用于测试特定参数组合的Costas环性能
    
    % 频率限幅设置
    freq_max_rad = 2*pi*freq_max/fs;
    freq_min_rad = -freq_max_rad;
    
    % 初始化变量
    N = length(signal);
    phase = 0;
    freq = 0;
    freq_history = zeros(1, N);
    
    % 环路滤波器系数计算
    K1 = 4 * damping * noise_bw / (1 + 2 * damping * noise_bw + noise_bw^2);
    K2 = 4 * noise_bw^2 / (1 + 2 * damping * noise_bw + noise_bw^2);
    
    % Costas环处理
    I_arm = zeros(1, N);
    Q_arm = zeros(1, N);
    error = zeros(1, N);
    
    for n = 1:N
        % 生成本地载波
        I_carrier = cos(2 * pi * f_carrier * n / fs + phase);
        Q_carrier = -sin(2 * pi * f_carrier * n / fs + phase);
        
        % I/Q解调
        I_arm(n) = signal(n) * I_carrier;
        Q_arm(n) = signal(n) * Q_carrier;
        
        % 相位检测器
        error(n) = sign(I_arm(n)) * Q_arm(n);
        
        % 环路滤波器
        freq = freq + K2 * error(n);
        freq = max(min(freq, freq_max_rad), freq_min_rad);
        
        phase = phase + freq + K1 * error(n);
        phase = mod(phase + pi, 2*pi) - pi;
        
        freq_history(n) = freq;
    end
    
    % 计算频率误差
    steady_state_start = floor(N * 0.7);
    avg_freq_radians = mean(freq_history(steady_state_start:end));
    freq_error = avg_freq_radians * fs / (2 * pi);
    
    % 计算SNR估计
    I_steady = I_arm(steady_state_start:end);
    Q_steady = Q_arm(steady_state_start:end);
    signal_power = mean(I_steady.^2);
    noise_power = mean(Q_steady.^2);
    snr_estimate = 10 * log10(signal_power / noise_power);
    snr_estimate = min(max(snr_estimate, 0), 40);
end 
```


### src\sync\square_law_sync.m

```matlab
function [freq_error, snr_estimate] = square_law_sync(signal, fs, f_carrier)
    % 平方变换法实现载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率
    %   f_carrier: 载波频率
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)
    
    % 平方变换
    squared_signal = signal.^2;
    
    % FFT分析
    N = length(squared_signal);
    fft_result = fft(squared_signal);
    fft_result = fftshift(fft_result);
    magnitude = abs(fft_result);
    freq = (-N/2:N/2-1) * (fs / N);
    
    % 寻找二倍频成分的峰值
    expected_freq = 2 * f_carrier;  % 二倍载波频率
    [~, center_idx] = min(abs(freq));  % 找到频谱中心点
    search_width = floor(N / 8);  % 搜索范围宽度
    search_start = max(1, center_idx + floor(expected_freq/(fs/N)) - search_width);
    search_end = min(N, center_idx + floor(expected_freq/(fs/N)) + search_width);
    search_range = search_start:search_end;
    
    [peak_value, local_peak_idx] = max(magnitude(search_range));
    peak_idx = search_start + local_peak_idx - 1;
    peak_freq = freq(peak_idx);
    
    % 计算频率误差（考虑二倍频的影响）
    freq_error = (peak_freq / 2) - f_carrier;
    
    % 改进的SNR估计
    % 使用峰值周围的平均功率作为信号功率
    signal_range = max(1, peak_idx - 2):min(N, peak_idx + 2);
    signal_power = mean(magnitude(signal_range).^2);
    
    % 排除峰值附近区域计算噪声功率
    exclude_range = max(1, peak_idx - 10):min(N, peak_idx + 10);
    noise_magnitude = magnitude;
    noise_magnitude(exclude_range) = [];
    noise_power = mean(noise_magnitude.^2);
    
    % 计算SNR
    snr_estimate = 10 * log10(signal_power / noise_power);
end 
```


### test\sync\test_sync_methods.m

```matlab
%{
载波同步方法测试脚本 - 第二版

测试目标：
1. 对比四种方法的性能：
   - 平方律法
   - 原始Costas环
   - 改进的Costas环
   - 多级同步器

2. 测试条件：
   - 频率偏移：0.5~50 Hz（扩大范围）
   - SNR：10~30 dB
   - 信号长度：10秒

3. 评估指标：
   - 频率估计精度
   - SNR估计准确度
   - 捕获时间
   - 计算复杂度

4. 可视化：
   - 频率估计误差随频偏变化
   - SNR估计误差分布
   - 同步过程动态特性
%}

function test_sync_methods()
    % 测试参数
    fs = 1000;          % 采样频率
    f_carrier = 100;    % 载波频率
    signal_length = 10; % 信号长度（秒）
    
    % 扩展测试条件
    freq_offsets = [0.5, 1, 2, 5, 10, 20, 30, 40, 50];  % Hz
    snrs = [10, 20, 30];  % dB
    
    % 创建结果文件
    fid = fopen('sync_results.txt', 'w');
    fprintf(fid, '载波同步系统测试结果（第二版）\n');
    fprintf(fid, '===================\n\n');
    
    % 创建性能统计结构
    stats = initialize_statistics();
    
    % 创建图形窗口
    figure('Name', '同步性能对比', 'Position', [100, 100, 1200, 800]);
    
    % 对每种测试条件进行测试
    for f_offset = freq_offsets
        for snr = snrs
            % 生成测试信号
            N = floor(fs * signal_length);  % 确保采样点数量正确
            t = (0:N-1)/fs;  % 生成对应的时间向量
            modulated_signal = cos(2*pi*(f_carrier + f_offset)*t);
            noisy_signal = awgn(modulated_signal, snr);
            
            % 测试各种方法
            tic;
            [freq_error1, snr_est1] = square_law_sync(noisy_signal, fs, f_carrier);
            time1 = toc;
            
            tic;
            [freq_error2, snr_est2] = costas_loop_sync(noisy_signal, fs, f_carrier);
            time2 = toc;
            
            tic;
            [freq_error3, snr_est3, debug3] = improved_costas_sync(noisy_signal, fs, f_carrier);
            time3 = toc;
            
            tic;
            [freq_error4, snr_est4, debug4] = multi_stage_costas_sync(noisy_signal, fs, f_carrier);
            time4 = toc;
            
            % 记录结果
            fprintf(fid, '测试条件:\n');
            fprintf(fid, '频率偏差: %.1f Hz\n', f_offset);
            fprintf(fid, 'SNR: %.0f dB\n\n', snr);
            
            % 记录各方法结果
            methods = {'平方变换法', '原始Costas环法', '改进Costas环法', '多级同步法'};
            freq_errors = [freq_error1, freq_error2, freq_error3, freq_error4];
            snr_ests = [snr_est1, snr_est2, snr_est3, snr_est4];
            times = [time1, time2, time3, time4];
            
            for i = 1:length(methods)
                fprintf(fid, '%s结果:\n', methods{i});
                fprintf(fid, '估计频率误差: %.2f Hz\n', freq_errors(i));
                fprintf(fid, '估计SNR: %.2f dB\n', snr_ests(i));
                freq_err_percent = abs((freq_errors(i)-f_offset)/f_offset)*100;
                fprintf(fid, '频率误差精度: %.2f%%\n', freq_err_percent);
                fprintf(fid, '处理时间: %.3f 秒\n\n', times(i));
                
                % 更新统计信息
                stats = update_statistics(stats, i, f_offset, snr, ...
                    freq_err_percent, abs(snr_ests(i)-snr), times(i));
            end
            
            fprintf(fid, '-------------------\n\n');
            
            % 绘制同步过程
            plot_sync_process(debug4, f_offset, snr, t);
        end
    end
    
    % 输出统计摘要
    print_statistics_summary(fid, stats, methods);
    
    fclose(fid);
    
    % 绘制性能对比图
    plot_performance_comparison(stats, freq_offsets, methods);
end

function stats = initialize_statistics()
    % 初始化统计结构
    stats = struct();
    stats.freq_errors = cell(4,1);
    stats.snr_errors = cell(4,1);
    stats.times = cell(4,1);
    for i = 1:4
        stats.freq_errors{i} = [];
        stats.snr_errors{i} = [];
        stats.times{i} = [];
    end
end

function stats = update_statistics(stats, method_idx, f_offset, snr, ...
    freq_error, snr_error, time)
    % 更新统计信息
    stats.freq_errors{method_idx}(end+1,:) = [f_offset, snr, freq_error];
    stats.snr_errors{method_idx}(end+1,:) = [f_offset, snr, snr_error];
    stats.times{method_idx}(end+1,:) = [f_offset, snr, time];
end

function print_statistics_summary(fid, stats, methods)
    % 输出统计摘要
    fprintf(fid, '\n性能统计摘要\n');
    fprintf(fid, '===================\n\n');
    
    for i = 1:length(methods)
        fprintf(fid, '%s:\n', methods{i});
        
        % 频率误差统计
        freq_errs = stats.freq_errors{i}(:,3);
        fprintf(fid, '频率误差统计:\n');
        fprintf(fid, '  平均值: %.2f%%\n', mean(freq_errs));
        fprintf(fid, '  中位数: %.2f%%\n', median(freq_errs));
        fprintf(fid, '  最大值: %.2f%%\n', max(freq_errs));
        fprintf(fid, '  最小值: %.2f%%\n', min(freq_errs));
        fprintf(fid, '  标准差: %.2f%%\n\n', std(freq_errs));
        
        % SNR误差统计
        snr_errs = stats.snr_errors{i}(:,3);
        fprintf(fid, 'SNR误差统计:\n');
        fprintf(fid, '  平均值: %.2f dB\n', mean(snr_errs));
        fprintf(fid, '  中位数: %.2f dB\n', median(snr_errs));
        fprintf(fid, '  最大值: %.2f dB\n', max(snr_errs));
        fprintf(fid, '  最小值: %.2f dB\n', min(snr_errs));
        fprintf(fid, '  标准差: %.2f dB\n\n', std(snr_errs));
        
        % 处理时间统计
        times = stats.times{i}(:,3);
        fprintf(fid, '处理时间统计:\n');
        fprintf(fid, '  平均值: %.3f 秒\n', mean(times));
        fprintf(fid, '  中位数: %.3f 秒\n', median(times));
        fprintf(fid, '  最大值: %.3f 秒\n', max(times));
        fprintf(fid, '  最小值: %.3f 秒\n\n', min(times));
    end
end

function plot_sync_process(debug_info, f_offset, snr, t)
    % 绘制同步过程
    clf;
    
    % 频率估计过程
    subplot(4,1,1);
    plot(t, debug_info.tracking_stage.freq_history);
    hold on;
    plot([t(1), t(end)], [f_offset, f_offset], 'r--');
    title(sprintf('频率估计过程 (offset=%.1fHz, SNR=%.0fdB)', f_offset, snr));
    xlabel('时间 (s)');
    ylabel('频率 (Hz)');
    legend('估计值', '实际值');
    grid on;
    
    % 相位估计过程
    subplot(4,1,2);
    plot(t, debug_info.tracking_stage.phase_history);
    title('相位估计过程');
    xlabel('时间 (s)');
    ylabel('相位 (rad)');
    grid on;
    
    % 带宽变化
    subplot(4,1,3);
    plot(t, debug_info.tracking_stage.bw_history);
    title('环路带宽变化');
    xlabel('时间 (s)');
    ylabel('带宽');
    grid on;
    
    % 误差信号
    subplot(4,1,4);
    plot(t, debug_info.tracking_stage.error_history);
    title('误差信号');
    xlabel('时间 (s)');
    ylabel('误差');
    grid on;
    
    % 添加多级同步信息
    annotation('textbox', [0.15, 0.95, 0.7, 0.05], ...
        'String', sprintf('多级同步过程: 粗估计=%.2fHz, 精细估计=%.2fHz', ...
        debug_info.coarse_stage.freq_error, ...
        debug_info.fine_stage.freq_error), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    
    drawnow;
    pause(0.1);
end

function plot_performance_comparison(stats, freq_offsets, methods)
    % 创建性能对比图
    figure('Name', '性能对比', 'Position', [100, 100, 1200, 800]);
    
    % 频率误差vs频率偏移
    subplot(2,2,1);
    for i = 1:length(methods)
        data = stats.freq_errors{i};
        for snr = unique(data(:,2))'
            mask = data(:,2) == snr;
            plot(data(mask,1), data(mask,3), 'o-', 'DisplayName', ...
                sprintf('%s (SNR=%ddB)', methods{i}, snr));
            hold on;
        end
    end
    xlabel('频率偏移 (Hz)');
    ylabel('频率误差 (%)');
    title('频率估计性能');
    grid on;
    legend('show');
    
    % SNR误差vs频率偏移
    subplot(2,2,2);
    for i = 1:length(methods)
        data = stats.snr_errors{i};
        for snr = unique(data(:,2))'
            mask = data(:,2) == snr;
            plot(data(mask,1), data(mask,3), 'o-', 'DisplayName', ...
                sprintf('%s (SNR=%ddB)', methods{i}, snr));
            hold on;
        end
    end
    xlabel('频率偏移 (Hz)');
    ylabel('SNR误差 (dB)');
    title('SNR估计性能');
    grid on;
    legend('show');
    
    % 处理时间vs频率偏移
    subplot(2,2,3);
    for i = 1:length(methods)
        data = stats.times{i};
        plot(data(:,1), data(:,3), 'o-', 'DisplayName', methods{i});
        hold on;
    end
    xlabel('频率偏移 (Hz)');
    ylabel('处理时间 (s)');
    title('计算复杂度');
    grid on;
    legend('show');
    
    % 箱线图比较
    subplot(2,2,4);
    % 重新组织数据用于箱线图
    all_errors = cell(1, length(methods));
    for i = 1:length(methods)
        all_errors{i} = stats.freq_errors{i}(:,3);
    end
    boxplot(cell2mat(all_errors), 'Labels', methods);
    ylabel('频率误差 (%)');
    title('方法间性能对比');
    grid on;
end 
```


## 测试结果

```
载波同步系统测试结果（第二版）
===================

测试条件:
频率偏差: 0.5 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 0.50 Hz
估计SNR: 23.02 dB
频率误差精度: 0.00%
处理时间: 1.163 秒

原始Costas环法结果:
估计频率误差: 0.50 Hz
估计SNR: 3.70 dB
频率误差精度: 0.84%
处理时间: 0.030 秒

改进Costas环法结果:
估计频率误差: 0.50 Hz
估计SNR: 20.81 dB
频率误差精度: 0.34%
处理时间: 0.679 秒

多级同步法结果:
估计频率误差: 0.50 Hz
估计SNR: 11.80 dB
频率误差精度: 0.19%
处理时间: 0.608 秒

-------------------

测试条件:
频率偏差: 0.5 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 0.50 Hz
估计SNR: 25.61 dB
频率误差精度: 0.00%
处理时间: 0.008 秒

原始Costas环法结果:
估计频率误差: 0.50 Hz
估计SNR: 4.42 dB
频率误差精度: 0.43%
处理时间: 0.009 秒

改进Costas环法结果:
估计频率误差: 0.50 Hz
估计SNR: 22.10 dB
频率误差精度: 0.36%
处理时间: 0.094 秒

多级同步法结果:
估计频率误差: 0.50 Hz
估计SNR: 12.48 dB
频率误差精度: 0.28%
处理时间: 0.335 秒

-------------------

测试条件:
频率偏差: 0.5 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 0.50 Hz
估计SNR: 25.96 dB
频率误差精度: 0.00%
处理时间: 0.003 秒

原始Costas环法结果:
估计频率误差: 0.50 Hz
估计SNR: 4.58 dB
频率误差精度: 0.04%
处理时间: 0.002 秒

改进Costas环法结果:
估计频率误差: 0.50 Hz
估计SNR: 22.21 dB
频率误差精度: 0.05%
处理时间: 0.024 秒

多级同步法结果:
估计频率误差: 0.50 Hz
估计SNR: 12.56 dB
频率误差精度: 0.26%
处理时间: 0.250 秒

-------------------

测试条件:
频率偏差: 1.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 1.00 Hz
估计SNR: 22.98 dB
频率误差精度: 0.00%
处理时间: 0.002 秒

原始Costas环法结果:
估计频率误差: 1.01 Hz
估计SNR: 3.51 dB
频率误差精度: 0.75%
处理时间: 0.003 秒

改进Costas环法结果:
估计频率误差: 1.00 Hz
估计SNR: 20.81 dB
频率误差精度: 0.08%
处理时间: 0.015 秒

多级同步法结果:
估计频率误差: 1.00 Hz
估计SNR: 11.77 dB
频率误差精度: 0.02%
处理时间: 0.278 秒

-------------------

测试条件:
频率偏差: 1.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 1.00 Hz
估计SNR: 25.63 dB
频率误差精度: 0.00%
处理时间: 0.006 秒

原始Costas环法结果:
估计频率误差: 1.00 Hz
估计SNR: 4.51 dB
频率误差精度: 0.01%
处理时间: 0.006 秒

改进Costas环法结果:
估计频率误差: 1.00 Hz
估计SNR: 22.04 dB
频率误差精度: 0.05%
处理时间: 0.025 秒

多级同步法结果:
估计频率误差: 1.00 Hz
估计SNR: 12.48 dB
频率误差精度: 0.01%
处理时间: 0.243 秒

-------------------

测试条件:
频率偏差: 1.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 1.00 Hz
估计SNR: 25.96 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 1.00 Hz
估计SNR: 4.58 dB
频率误差精度: 0.06%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 1.00 Hz
估计SNR: 22.21 dB
频率误差精度: 0.08%
处理时间: 0.011 秒

多级同步法结果:
估计频率误差: 1.00 Hz
估计SNR: 12.57 dB
频率误差精度: 0.15%
处理时间: 0.275 秒

-------------------

测试条件:
频率偏差: 2.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 2.00 Hz
估计SNR: 22.81 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 2.00 Hz
估计SNR: 3.52 dB
频率误差精度: 0.18%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 2.00 Hz
估计SNR: 20.81 dB
频率误差精度: 0.10%
处理时间: 0.011 秒

多级同步法结果:
估计频率误差: 2.00 Hz
估计SNR: 11.77 dB
频率误差精度: 0.03%
处理时间: 0.266 秒

-------------------

测试条件:
频率偏差: 2.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 2.00 Hz
估计SNR: 25.64 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 2.00 Hz
估计SNR: 4.57 dB
频率误差精度: 0.02%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 2.00 Hz
估计SNR: 22.01 dB
频率误差精度: 0.02%
处理时间: 0.011 秒

多级同步法结果:
估计频率误差: 2.00 Hz
估计SNR: 12.49 dB
频率误差精度: 0.01%
处理时间: 0.276 秒

-------------------

测试条件:
频率偏差: 2.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 2.00 Hz
估计SNR: 25.97 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 2.00 Hz
估计SNR: 4.58 dB
频率误差精度: 0.02%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 2.00 Hz
估计SNR: 22.22 dB
频率误差精度: 0.02%
处理时间: 0.011 秒

多级同步法结果:
估计频率误差: 2.00 Hz
估计SNR: 12.57 dB
频率误差精度: 0.09%
处理时间: 0.246 秒

-------------------

测试条件:
频率偏差: 5.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 5.00 Hz
估计SNR: 22.87 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.51 Hz
估计SNR: 0.00 dB
频率误差精度: 110.14%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 5.00 Hz
估计SNR: 20.72 dB
频率误差精度: 0.08%
处理时间: 0.011 秒

多级同步法结果:
估计频率误差: 5.00 Hz
估计SNR: 11.68 dB
频率误差精度: 0.02%
处理时间: 0.283 秒

-------------------

测试条件:
频率偏差: 5.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 5.00 Hz
估计SNR: 25.60 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.01%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 5.00 Hz
估计SNR: 22.06 dB
频率误差精度: 0.00%
处理时间: 0.010 秒

多级同步法结果:
估计频率误差: 5.00 Hz
估计SNR: 12.48 dB
频率误差精度: 0.01%
处理时间: 0.244 秒

-------------------

测试条件:
频率偏差: 5.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 5.00 Hz
估计SNR: 25.97 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.02%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 5.00 Hz
估计SNR: 22.19 dB
频率误差精度: 0.01%
处理时间: 0.010 秒

多级同步法结果:
估计频率误差: 5.00 Hz
估计SNR: 12.57 dB
频率误差精度: 0.02%
处理时间: 0.234 秒

-------------------

测试条件:
频率偏差: 10.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 10.00 Hz
估计SNR: 22.83 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 99.98%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 9.81 Hz
估计SNR: 20.79 dB
频率误差精度: 1.86%
处理时间: 0.010 秒

多级同步法结果:
估计频率误差: 9.99 Hz
估计SNR: 11.67 dB
频率误差精度: 0.05%
处理时间: 0.205 秒

-------------------

测试条件:
频率偏差: 10.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 10.00 Hz
估计SNR: 25.61 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 99.98%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 9.90 Hz
估计SNR: 22.08 dB
频率误差精度: 0.98%
处理时间: 0.011 秒

多级同步法结果:
估计频率误差: 10.00 Hz
估计SNR: 12.49 dB
频率误差精度: 0.00%
处理时间: 0.195 秒

-------------------

测试条件:
频率偏差: 10.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 10.00 Hz
估计SNR: 25.97 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 9.92 Hz
估计SNR: 22.20 dB
频率误差精度: 0.80%
处理时间: 0.008 秒

多级同步法结果:
估计频率误差: 10.00 Hz
估计SNR: 12.58 dB
频率误差精度: 0.02%
处理时间: 0.192 秒

-------------------

测试条件:
频率偏差: 20.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 20.00 Hz
估计SNR: 22.85 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.01 Hz
估计SNR: 0.00 dB
频率误差精度: 100.06%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 8.14 Hz
估计SNR: 0.70 dB
频率误差精度: 59.28%
处理时间: 0.010 秒

多级同步法结果:
估计频率误差: 20.01 Hz
估计SNR: 12.26 dB
频率误差精度: 0.04%
处理时间: 0.234 秒

-------------------

测试条件:
频率偏差: 20.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 20.00 Hz
估计SNR: 25.59 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 99.99%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 8.23 Hz
估计SNR: 1.63 dB
频率误差精度: 58.87%
处理时间: 0.009 秒

多级同步法结果:
估计频率误差: 20.00 Hz
估计SNR: 13.03 dB
频率误差精度: 0.00%
处理时间: 0.194 秒

-------------------

测试条件:
频率偏差: 20.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 20.00 Hz
估计SNR: 25.97 dB
频率误差精度: 0.00%
处理时间: 0.000 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 8.34 Hz
估计SNR: 2.80 dB
频率误差精度: 58.32%
处理时间: 0.008 秒

多级同步法结果:
估计频率误差: 20.00 Hz
估计SNR: 13.13 dB
频率误差精度: 0.01%
处理时间: 0.174 秒

-------------------

测试条件:
频率偏差: 30.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 30.00 Hz
估计SNR: 22.87 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 8.47 Hz
估计SNR: 0.00 dB
频率误差精度: 71.76%
处理时间: 0.009 秒

多级同步法结果:
估计频率误差: 29.99 Hz
估计SNR: 12.62 dB
频率误差精度: 0.05%
处理时间: 0.171 秒

-------------------

测试条件:
频率偏差: 30.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 30.00 Hz
估计SNR: 25.59 dB
频率误差精度: 0.00%
处理时间: 0.004 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.005 秒

改进Costas环法结果:
估计频率误差: 8.50 Hz
估计SNR: 0.00 dB
频率误差精度: 71.67%
处理时间: 0.034 秒

多级同步法结果:
估计频率误差: 30.01 Hz
估计SNR: 13.62 dB
频率误差精度: 0.02%
处理时间: 0.247 秒

-------------------

测试条件:
频率偏差: 30.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 30.00 Hz
估计SNR: 25.96 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 8.73 Hz
估计SNR: 0.00 dB
频率误差精度: 70.90%
处理时间: 0.011 秒

多级同步法结果:
估计频率误差: 30.00 Hz
估计SNR: 13.72 dB
频率误差精度: 0.01%
处理时间: 0.220 秒

-------------------

测试条件:
频率偏差: 40.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 40.00 Hz
估计SNR: 22.98 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.01%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 8.66 Hz
估计SNR: 0.00 dB
频率误差精度: 78.35%
处理时间: 0.009 秒

多级同步法结果:
估计频率误差: 39.99 Hz
估计SNR: 13.14 dB
频率误差精度: 0.03%
处理时间: 0.191 秒

-------------------

测试条件:
频率偏差: 40.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 40.00 Hz
估计SNR: 25.60 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 8.77 Hz
估计SNR: 0.00 dB
频率误差精度: 78.08%
处理时间: 0.009 秒

多级同步法结果:
估计频率误差: 40.00 Hz
估计SNR: 14.24 dB
频率误差精度: 0.00%
处理时间: 0.175 秒

-------------------

测试条件:
频率偏差: 40.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 40.00 Hz
估计SNR: 25.98 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 8.75 Hz
估计SNR: 0.00 dB
频率误差精度: 78.12%
处理时间: 0.010 秒

多级同步法结果:
估计频率误差: 40.00 Hz
估计SNR: 14.36 dB
频率误差精度: 0.00%
处理时间: 0.191 秒

-------------------

测试条件:
频率偏差: 50.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 50.00 Hz
估计SNR: 22.76 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.01%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 8.75 Hz
估计SNR: 0.00 dB
频率误差精度: 82.50%
处理时间: 0.009 秒

多级同步法结果:
估计频率误差: 49.93 Hz
估计SNR: 13.54 dB
频率误差精度: 0.14%
处理时间: 0.187 秒

-------------------

测试条件:
频率偏差: 50.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 50.00 Hz
估计SNR: 25.60 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.008 秒

多级同步法结果:
估计频率误差: 49.96 Hz
估计SNR: 14.87 dB
频率误差精度: 0.07%
处理时间: 0.200 秒

-------------------

测试条件:
频率偏差: 50.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 50.00 Hz
估计SNR: 25.97 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.011 秒

多级同步法结果:
估计频率误差: 49.97 Hz
估计SNR: 15.05 dB
频率误差精度: 0.06%
处理时间: 0.222 秒

-------------------


性能统计摘要
===================

平方变换法:
频率误差统计:
  平均值: 0.00%
  中位数: 0.00%
  最大值: 0.00%
  最小值: 0.00%
  标准差: 0.00%

SNR误差统计:
  平均值: 7.51 dB
  中位数: 5.60 dB
  最大值: 13.02 dB
  最小值: 4.02 dB
  标准差: 3.93 dB

处理时间统计:
  平均值: 0.044 秒
  中位数: 0.001 秒
  最大值: 1.163 秒
  最小值: 0.000 秒

原始Costas环法:
频率误差统计:
  平均值: 67.13%
  中位数: 100.00%
  最大值: 110.14%
  最小值: 0.01%
  标准差: 48.22%

SNR误差统计:
  平均值: 18.59 dB
  中位数: 20.00 dB
  最大值: 30.00 dB
  最小值: 6.30 dB
  标准差: 8.43 dB

处理时间统计:
  平均值: 0.003 秒
  中位数: 0.001 秒
  最大值: 0.030 秒
  最小值: 0.001 秒

改进Costas环法:
频率误差统计:
  平均值: 33.80%
  中位数: 0.98%
  最大值: 100.00%
  最小值: 0.00%
  标准差: 39.22%

SNR误差统计:
  平均值: 12.52 dB
  中位数: 10.00 dB
  最大值: 30.00 dB
  最小值: 2.01 dB
  标准差: 8.85 dB

处理时间统计:
  平均值: 0.040 秒
  中位数: 0.011 秒
  最大值: 0.679 秒
  最小值: 0.008 秒

多级同步法:
频率误差统计:
  平均值: 0.06%
  中位数: 0.02%
  最大值: 0.28%
  最小值: 0.00%
  标准差: 0.08%

SNR误差统计:
  平均值: 8.63 dB
  中位数: 7.51 dB
  最大值: 17.44 dB
  最小值: 1.67 dB
  标准差: 6.23 dB

处理时间统计:
  平均值: 0.242 秒
  中位数: 0.234 秒
  最大值: 0.608 秒
  最小值: 0.171 秒

```

