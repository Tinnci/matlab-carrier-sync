# MATLAB同步算法项目文档

## 目录结构

### 文件树形图

```
Folder PATH listing of E:\sync
Volume serial number is .

├───docs
│   ├───matlab_files.md
├───src
│   ├───optimization
│   │   ├───optimize_costas_params.m
│   ├───sync
│   │   ├───calculate_loop_coefficients.m
│   │   ├───costas_loop_sync.m
│   │   ├───improved_costas_sync.m
│   │   ├───multi_stage_costas_sync.m
│   │   ├───particle_filter_sync.m
│   │   ├───pll_sync.m
│   │   ├───square_law_sync.m
├───test
│   ├───sync
│   │   ├───test_sync_methods.m
├───.gitignore
├───concat_matlab_files.ps1
├───main.m
├───README.md
├───run_matlab_script.ps1
```

### 目录列表

```
- docs
- src
  - optimization
  - sync
- test
  - sync
```

## 源代码文件

- main.m
- src\optimization\optimize_costas_params.m
- src\sync\calculate_loop_coefficients.m
- src\sync\costas_loop_sync.m
- src\sync\improved_costas_sync.m
- src\sync\multi_stage_costas_sync.m
- src\sync\particle_filter_sync.m
- src\sync\pll_sync.m
- src\sync\square_law_sync.m
- test\sync\test_sync_methods.m

---


## 源代码内容


### main.m

```matlab
% main.m
% 主程序文件
% 运行载波同步系统测试

% 获取当前脚本所在目录
current_dir = fileparts(mfilename('fullpath'));

% 添加所需路径
addpath(fullfile(current_dir, 'src', 'sync'));
addpath(fullfile(current_dir, 'src', 'optimization'));
addpath(fullfile(current_dir, 'test', 'sync'));

% 清理工作空间
clear;
clc;

% 运行测试
test_sync_methods(); 
```


### src\optimization\optimize_costas_params.m

```matlab
% optimize_costas_params.m
%{
优化Costas环参数

问题描述：
1. 当前Costas环存在以下问题：
   - 小频偏（0.5~2 Hz）时性能很好
   - 大频偏（5~150 Hz）时性能不稳定
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
1. 使用并行网格搜索遍历参数空间
2. 对每组参数进行Monte Carlo测试
3. 综合评估频率误差和SNR误差
4. 选择最优参数组合
%}

function best_params = optimize_costas_params()
    % 优化Costas环参数
    % 输出:
    %   best_params: 结构体，包含最优参数

    % 测试参数设置
    freq_offsets = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150];  % Hz 扩展到150Hz
    snrs = [10, 20, 30];  % dB
    fs = 1000;  % Hz
    f_carrier = 100;  % Hz
    signal_length = 10;  % 秒
    monte_carlo_runs = 10;  % 每组参数测试次数

    % 参数搜索范围
    noise_bw_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3];  % 扩展到0.3
    damping_range = [0.4, 0.5, 0.6, 0.707, 1.0];  % 包含更低的阻尼系数
    freq_max_range = [10, 25, 50, 100, 150];  % 扩展到150Hz

    % 使用 ndgrid 生成所有组合
    [A, B, C] = ndgrid(noise_bw_range, damping_range, freq_max_range);
    param_combinations = [A(:), B(:), C(:)];
    num_combinations = size(param_combinations, 1);

    % 初始化评分数组
    scores = zeros(num_combinations, 1);

    % 创建结果文件
    fid = fopen('optimization_results.txt', 'w');
    fprintf(fid, 'Costas环参数优化结果\n');
    fprintf(fid, '===================\n\n');

    % 开启并行池
    pool = gcp('nocreate');
    if isempty(pool)
        parpool;
    end

    % 并行参数搜索
    parfor idx = 1:num_combinations
        noise_bw = param_combinations(idx, 1);
        damping = param_combinations(idx, 2);
        freq_max = param_combinations(idx, 3);

        % 当前参数组合的性能统计
        freq_errors = [];
        snr_errors = [];

        % 对每种测试条件进行多次Monte Carlo测试
        for f_offset = freq_offsets
            for snr = snrs
                % 生成预先的信号模板
                t = 0:1/fs:signal_length;
                modulated_signal = cos(2*pi*(f_carrier + f_offset)*t);
                % 使用相同的随机种子确保信号一致性
                rng(0);  
                noisy_signal = awgn(modulated_signal, snr, 'measured');

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

        % 计算性能得分
        % 频率误差权重更大，因为这是主要优化目标
        freq_score = -mean(freq_errors);  % 负值因为误差越小越好
        snr_score = -mean(snr_errors);
        total_score = freq_score * 0.8 + snr_score * 0.2;

        % 存储得分
        scores(idx) = total_score;
    end

    % 将得分添加到 param_combinations
    param_combinations(:,4) = scores;

    % 找到最优参数
    [~, best_idx] = max(param_combinations(:,4));
    best_params.noise_bw = param_combinations(best_idx, 1);
    best_params.damping = param_combinations(best_idx, 2);
    best_params.freq_max = param_combinations(best_idx, 3);
    best_params.score = param_combinations(best_idx, 4);

    % 记录所有参数组合的结果
    for idx = 1:num_combinations
        noise_bw = param_combinations(idx, 1);
        damping = param_combinations(idx, 2);
        freq_max = param_combinations(idx, 3);
        total_score = param_combinations(idx, 4);

        fprintf(fid, '参数组合:\n');
        fprintf(fid, 'noise_bw: %.3f\n', noise_bw);
        fprintf(fid, 'damping: %.3f\n', damping);
        fprintf(fid, 'freq_max: %.1f Hz\n', freq_max);
        fprintf(fid, '总得分: %.2f\n\n', total_score);
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
    [K1, K2] = calculate_loop_coefficients(noise_bw, damping);

    % 初始化I/Q分支
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

        % 相位检测器
        error(n) = sign(I_arm(n)) * Q_arm(n);

        % 环路滤波器
        freq = freq + K2 * error(n);
        freq = max(min(freq, freq_max_rad), freq_min_rad);

        % 相位更新
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


### src\sync\calculate_loop_coefficients.m

```matlab
% calculate_loop_coefficients.m
function [K1, K2] = calculate_loop_coefficients(noise_bw, damping)
    % 计算环路滤波器系数
    % 输入参数:
    %   noise_bw: 噪声带宽
    %   damping: 阻尼系数
    % 输出参数:
    %   K1, K2: 环路滤波器系数

    K1 = 4 * damping * noise_bw / (1 + 2 * damping * noise_bw + noise_bw^2);
    K2 = 4 * noise_bw^2 / (1 + 2 * damping * noise_bw + noise_bw^2);
end
```


### src\sync\costas_loop_sync.m

```matlab
% costas_loop_sync.m
function [freq_error, snr_estimate] = costas_loop_sync(signal, fs, f_carrier, noise_bw, damping, freq_max)
    % Costas环法实现载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率 (Hz)
    %   f_carrier: 载波频率 (Hz)
    %   noise_bw: 噪声带宽
    %   damping: 阻尼系数
    %   freq_max: 最大频率偏移 (Hz)
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)

    % 频率限幅设置
    freq_max_rad = 2*pi*freq_max/fs;
    freq_min_rad = -freq_max_rad;

    % 初始化变量
    N = length(signal);
    phase = 0;
    freq = 0;
    freq_history = zeros(1, N);

    % 环路滤波器系数计算
    [K1, K2] = calculate_loop_coefficients(noise_bw, damping);

    % 初始化I/Q分支
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

        % 相位检测器：使用符号判决增强抗噪性能
        error(n) = sign(I_arm(n)) * Q_arm(n);

        % 环路滤波器
        freq = freq + K2 * error(n);
        % 频率限幅
        freq = max(min(freq, freq_max_rad), freq_min_rad);

        % 相位更新
        phase = phase + freq + K1 * error(n);
        % 相位归一化到[-pi, pi]
        phase = mod(phase + pi, 2*pi) - pi;

        % 记录历史
        freq_history(n) = freq;
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
end
```


### src\sync\improved_costas_sync.m

```matlab
% improved_costas_sync.m
function [freq_error, snr_estimate, debug_info] = improved_costas_sync(signal, fs, f_carrier, noise_bw, damping, freq_max)
    % 改进的Costas环载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率 (Hz)
    %   f_carrier: 载波频率 (Hz)
    %   noise_bw: 噪声带宽
    %   damping: 阻尼系数
    %   freq_max: 最大频率偏移 (Hz)
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)
    %   debug_info: 调试信息结构体

    % 频率限幅设置
    freq_max_rad = 2*pi*freq_max/fs;
    freq_min_rad = -freq_max_rad;

    % 初始化变量
    N = length(signal);
    phase = 0;
    freq = 0;
    freq_history = zeros(1, N);
    phase_history = zeros(1, N);

    % 环路滤波器系数计算
    [K1, K2] = calculate_loop_coefficients(noise_bw, damping);

    % 初始化I/Q分支
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

        % 改进的相位检测器（arctan）
        error(n) = improved_phase_detector(I_arm(n), Q_arm(n));

        % 环路滤波器
        freq = freq + K2 * error(n);
        % 频率限幅
        freq = max(min(freq, freq_max_rad), freq_min_rad);

        % 相位更新
        phase = phase + freq + K1 * error(n);
        % 相位归一化到[-pi, pi]
        phase = mod(phase + pi, 2*pi) - pi;

        % 记录历史
        freq_history(n) = freq;
        phase_history(n) = phase;
    end

    % 使用稳态数据计算最终结果
    [freq_error, snr_estimate] = calculate_final_estimates_tracked(freq_history, I_arm, Q_arm, fs, f_carrier);

    % 收集调试信息，并添加 freq_error 字段
    debug_info = struct(...
        'freq_history', freq_history * fs / (2 * pi), ...
        'phase_history', phase_history, ...
        'error_signal', error, ...
        'freq_error', freq_error);  % 添加 freq_error 字段
end

function error = improved_phase_detector(I, Q)
    % 改进的相位检测器
    error = atan2(Q, I);

    % 加入软判决
    amplitude = sqrt(I^2 + Q^2);
    confidence = amplitude .* tanh(amplitude);  % 使用tanh限制误差幅度
    error = error .* confidence;
end

function [freq_error, snr_estimate] = calculate_final_estimates_tracked(freq_history, I_arm, Q_arm, fs, f_carrier)
    % 计算最终估计值
    N = length(freq_history);
    steady_state_start = floor(N * 0.6);  % 使用后40%的数据

    % 频率误差估计（使用中值滤波去除异常值）
    window_size = 101;
    if steady_state_start + window_size -1 > N
        window_size = N - steady_state_start +1;
    end
    freq_smooth = medfilt1(freq_history(steady_state_start:end), window_size);
    avg_freq_radians = mean(freq_smooth);
    freq_error = avg_freq_radians * fs / (2 * pi);

    % 改进的SNR估计
    segment_length = floor(fs / 5);  % 0.2秒segments
    num_segments = floor((N - steady_state_start + 1) / segment_length);
    snr_estimates = zeros(1, num_segments);

    for i = 1:num_segments
        start_idx = steady_state_start + (i-1)*segment_length;
        end_idx = start_idx + segment_length -1;
        if end_idx > N
            end_idx = N;
        end
        segment = I_arm(start_idx:end_idx) + 1j * Q_arm(start_idx:end_idx);

        % 使用Welch方法估计功率谱
        [pxx, f] = pwelch(segment, [], [], [], fs);

        % 自适应信号带宽检测
        center_freq = f_carrier + freq_error;
        signal_band = abs(f - center_freq) <= max(5, abs(freq_error)/10);

        signal_power = mean(pxx(signal_band));
        noise_power = mean(pxx(~signal_band));
        snr_estimates(i) = 10 * log10(signal_power / noise_power);
    end

    % 去除异常值后平均
    snr_estimate = median(snr_estimates);
    snr_estimate = min(max(snr_estimate, 0), 40);  % 限制范围
end
```


### src\sync\multi_stage_costas_sync.m

```matlab
% multi_stage_costas_sync.m
function [freq_error, snr_estimate, debug_info] = multi_stage_costas_sync(signal, fs, f_carrier, noise_bw, damping, freq_max)
    % 多级Costas环同步器
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率 (Hz)
    %   f_carrier: 载波频率 (Hz)
    %   noise_bw: 噪声带宽 (优化后的)
    %   damping: 阻尼系数 (优化后的)
    %   freq_max: 最大频率偏移 (Hz) (优化后的)
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)
    %   debug_info: 调试信息结构体

    % 第一级：FFT粗搜索
    [coarse_freq_error, initial_snr] = wide_range_fft_search(signal, fs, f_carrier, 2^nextpow2(length(signal)), 200);  % 扩展到±200Hz

    % 第二级：分段精细搜索
    [refined_freq_error, refined_snr] = fine_grid_search(signal, fs, f_carrier, coarse_freq_error, 50);  % 扩展到±50Hz

    % 第三级：改进的Costas环精确跟踪（使用优化后的参数）
    [final_freq_error, final_snr, tracking_info] = improved_costas_sync(...
        signal, fs, f_carrier, noise_bw, damping, freq_max);  % 使用优化后的参数

    % 返回最终结果
    freq_error = final_freq_error;
    snr_estimate = final_snr;

    % 收集调试信息
    debug_info = struct(...
        'coarse_stage', struct('freq_error', coarse_freq_error, 'snr', initial_snr), ...
        'fine_stage', struct('freq_error', refined_freq_error, 'snr', refined_snr), ...
        'tracking_stage', tracking_info);
end

function [freq_error, snr] = wide_range_fft_search(signal, fs, f_carrier, fft_size, search_range)
    % 宽范围FFT搜索
    % 使用多个窗口减少频谱泄漏，并进行峰值搜索

    % 使用多个窗口减少频谱泄漏
    window_types = {
        @hamming,
        @hanning,
        @blackman
    };

    freq = (-fft_size/2:fft_size/2-1)*(fs / fft_size);
    search_width_bins = round(search_range * fft_size / fs);
    [~, carrier_bin] = min(abs(freq - f_carrier));
    search_start = carrier_bin - search_width_bins;
    search_end = carrier_bin + search_width_bins;
    search_start = max(1, search_start);
    search_end = min(fft_size, search_end);
    search_indices = search_start:search_end;

    % 多窗口平均频谱
    avg_spectrum = zeros(1, fft_size);
    for i = 1:length(window_types)
        window = window_types{i}(length(signal))';
        windowed_signal = signal .* window;
        spectrum = fftshift(abs(fft(windowed_signal, fft_size)));
        avg_spectrum = avg_spectrum + spectrum / length(window_types);
    end

    % 在扩展后的搜索范围内搜索峰值
    [~, max_idx_local] = max(avg_spectrum(search_indices));
    peak_freq = freq(search_indices(max_idx_local));

    % 使用抛物线插值提高频率分辨率
    if max_idx_local > 1 && max_idx_local < length(search_indices)
        alpha = avg_spectrum(search_indices(max_idx_local-1));
        beta = avg_spectrum(search_indices(max_idx_local));
        gamma = avg_spectrum(search_indices(max_idx_local+1));
        offset = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma);
        peak_freq = peak_freq + offset * fs / fft_size;
    end

    freq_error = peak_freq - f_carrier;

    % 改进的SNR估计
    signal_band = abs(freq - peak_freq) <= 5;
    signal_power = mean(avg_spectrum(signal_band).^2);
    noise_power = mean(avg_spectrum(~signal_band).^2);
    snr = 10 * log10(signal_power / noise_power);
end

function [freq_error, snr] = fine_grid_search(signal, fs, f_carrier, coarse_error, search_width)
    % 分段精细搜索
    % 在粗搜索结果附近进行精细网格搜索

    % 搜索参数
    step_size = 0.5;    % Hz
    search_freqs = (coarse_error - search_width):step_size:(coarse_error + search_width);
    correlation_power = zeros(size(search_freqs));

    % 分段处理以减少计算量
    segment_length = floor(fs/10);  % 0.1秒segments
    num_segments = floor(length(signal)/segment_length);

    % 对每个频率假设计算相关性
    for i = 1:length(search_freqs)
        f_test = f_carrier + search_freqs(i);
        total_power = 0;

        % 分段处理
        for seg = 1:num_segments
            idx = (seg-1)*segment_length + (1:segment_length);
            t = (idx-1)/fs;
            test_signal = exp(-1j*2*pi*f_test*t);
            segment_correlation = abs(sum(signal(idx).*test_signal));
            total_power = total_power + segment_correlation;
        end

        correlation_power(i) = total_power / num_segments;
    end

    % 找到最佳匹配
    [~, max_idx] = max(correlation_power);
    freq_error = search_freqs(max_idx);

    % 计算SNR
    peak_power = correlation_power(max_idx);
    noise_indices = [1:max_idx-5, max_idx+5:length(correlation_power)];
    if isempty(noise_indices)
        noise_power = 1e-6;  % 避免除以零
    else
        noise_power = mean(correlation_power(noise_indices));
    end
    snr = 10 * log10(peak_power / noise_power);
end
```


### src\sync\particle_filter_sync.m

```matlab
% particle_filter_sync.m
function [freq_error, snr_estimate, debug_info] = particle_filter_sync(signal, fs, f_carrier, num_particles, freq_max)
    % 基于粒子滤波器的载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率 (Hz)
    %   f_carrier: 载波频率 (Hz)
    %   num_particles: 粒子数量
    %   freq_max: 最大频率偏移 (Hz)
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)
    %   debug_info: 调试信息结构体

    % 初始化粒子
    particles = f_carrier + (rand(1, num_particles) - 0.5) * 2 * freq_max;
    weights = ones(1, num_particles) / num_particles;

    % 初始化变量
    N = length(signal);
    freq_history = zeros(1, N);

    for n = 1:N
        % 生成本地载波
        t = (n-1)/fs;
        I_carrier = cos(2 * pi * particles * t);
        Q_carrier = -sin(2 * pi * particles * t);

        % I/Q解调
        I_arm = signal(n) * I_carrier;
        Q_arm = signal(n) * Q_carrier;

        % 计算观测
        obs = atan2(Q_arm, I_arm);

        % 计算权重
        weights = weights .* exp(-(obs).^2 / (2*(0.1)^2));
        weights = weights / sum(weights);

        % 重采样
        indices = resample_particles(weights);
        particles = particles(indices);
        weights = ones(1, num_particles) / num_particles;

        % 估计频率
        freq_history(n) = mean(particles);
    end

    % 计算频率误差
    steady_state_start = floor(N * 0.7);
    avg_freq = mean(freq_history(steady_state_start:end));
    freq_error = avg_freq - f_carrier;

    % 计算SNR估计
    I_steady = signal(steady_state_start:end) .* cos(2*pi*f_carrier*(steady_state_start:N)/fs);
    Q_steady = signal(steady_state_start:end) .* -sin(2*pi*f_carrier*(steady_state_start:N)/fs);
    signal_power = mean(I_steady.^2);
    noise_power = mean(Q_steady.^2);
    snr_estimate = 10 * log10(signal_power / noise_power);
    snr_estimate = min(max(snr_estimate, 0), 40);

    % 收集调试信息
    debug_info = struct(...
        'freq_history', freq_history, ...
        'particles', particles, ...
        'weights', weights, ...
        'freq_error', freq_error);
end

function indices = resample_particles(weights)
    % 重采样粒子
    % 输入参数:
    %   weights: 粒子权重
    % 输出参数:
    %   indices: 重采样后的粒子索引

    cumulative_sum = cumsum(weights);
    cumulative_sum(end) = 1.0; % 避免精度问题
    rand_vals = rand(1, length(weights));
    indices = arrayfun(@(rv) find(cumulative_sum >= rv, 1), rand_vals);
end
```


### src\sync\pll_sync.m

```matlab
% pll_sync.m
function [freq_error, snr_estimate, debug_info] = pll_sync(signal, fs, f_carrier, loop_bw, damping, freq_max)
    % 基于锁相环（PLL）的载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率 (Hz)
    %   f_carrier: 载波频率 (Hz)
    %   loop_bw: 环路带宽
    %   damping: 阻尼系数
    %   freq_max: 最大频率偏移 (Hz)
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)
    %   debug_info: 调试信息结构体

    % 初始化变量
    N = length(signal);
    phase = 0;
    freq = 0;
    freq_history = zeros(1, N);

    % 计算环路滤波器系数
    [Kp, Ki] = calculate_pll_coefficients(loop_bw, damping, fs);

    % 初始化I/Q分支
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

        % 相位检测器
        error(n) = I_arm(n) * Q_arm(n);  % 基本PLL相位误差

        % 环路滤波器
        freq = freq + Ki * error(n);
        freq = max(min(freq, freq_max), -freq_max);  % 频率限幅

        % 相位更新
        phase = phase + (Kp * error(n)) + freq;
        phase = mod(phase + pi, 2*pi) - pi;

        freq_history(n) = freq;
    end

    % 计算频率误差
    steady_state_start = floor(N * 0.7);
    avg_freq = mean(freq_history(steady_state_start:end));
    freq_error = avg_freq;

    % 计算SNR估计
    I_steady = I_arm(steady_state_start:end);
    Q_steady = Q_arm(steady_state_start:end);
    signal_power = mean(I_steady.^2);
    noise_power = mean(Q_steady.^2);
    snr_estimate = 10 * log10(signal_power / noise_power);
    snr_estimate = min(max(snr_estimate, 0), 40);

    % 收集调试信息
    debug_info = struct(...
        'freq_history', freq_history, ...
        'phase_history', phase, ...
        'error_signal', error, ...
        'freq_error', freq_error);
end

function [Kp, Ki] = calculate_pll_coefficients(loop_bw, damping, fs)
    % 计算PLL的比例和积分增益
    % 输入参数:
    %   loop_bw: 环路带宽
    %   damping: 阻尼系数
    %   fs: 采样频率 (Hz)
    % 输出参数:
    %   Kp: 比例增益
    %   Ki: 积分增益

    theta = loop_bw / fs;
    Kp = (2 * damping * theta) / (1 + 2 * damping * theta + theta^2);
    Ki = (theta^2) / (1 + 2 * damping * theta + theta^2);
end
```


### src\sync\square_law_sync.m

```matlab
% square_law_sync.m
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
    fft_result = fft(squared_signal, 2^nextpow2(N));
    fft_result = fftshift(fft_result);
    magnitude = abs(fft_result);
    freq = (-length(fft_result)/2:length(fft_result)/2-1) * (fs / length(fft_result));

    % 寻找二倍频成分的峰值
    expected_freq = 2 * f_carrier;  % 二倍载波频率
    [~, center_idx] = min(abs(freq - expected_freq));  % 找到接近二倍频的频谱点
    search_width = floor(length(fft_result) / 16);  % 搜索范围宽度
    search_start = max(1, center_idx - search_width);
    search_end = min(length(fft_result), center_idx + search_width);
    search_range = search_start:search_end;

    [peak_value, local_peak_idx] = max(magnitude(search_range));
    peak_idx = search_start + local_peak_idx - 1;
    peak_freq = freq(peak_idx);

    % 计算频率误差（考虑二倍频的影响）
    freq_error = (peak_freq / 2) - f_carrier;

    % 改进的SNR估计
    % 使用峰值周围的平均功率作为信号功率
    signal_range = max(1, peak_idx - 2):min(length(freq), peak_idx + 2);
    signal_power = mean(magnitude(signal_range).^2);

    % 排除峰值附近区域计算噪声功率
    noise_magnitude = magnitude;
    noise_magnitude(signal_range) = [];
    noise_power = mean(noise_magnitude.^2);

    % 计算SNR
    snr_estimate = 10 * log10(signal_power / noise_power);
    snr_estimate = min(max(snr_estimate, 0), 40);  % 限制范围
end 
```


### test\sync\test_sync_methods.m

```matlab
% test_sync_methods.m
function test_sync_methods()
    % 测试参数
    fs = 1000;          % 采样频率
    f_carrier = 100;    % 载波频率
    signal_length = 10; % 信号长度（秒）

    % 扩展测试条件
    freq_offsets = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150];  % 扩展到150Hz
    snrs = [10, 20, 30];  % dB

    % 调制方式
    modulation_types = {'AM', 'BPSK', 'QPSK', '16-QAM'};

    % 创建结果文件
    fid = fopen('sync_results.txt', 'w');
    fprintf(fid, '载波同步系统测试结果（第三版）\n');
    fprintf(fid, '===================\n\n');

    % 运行参数优化并获取最优参数
    fprintf('开始参数优化...\n');
    best_params = optimize_costas_params();  % 确保您有此函数
    fprintf('参数优化完成。\n\n');

    % 创建性能统计结构
    stats = initialize_statistics();

    % 获取中文显示名称
    display_names = stats.method_display_names;

    % 创建图形窗口
    figure('Name', '同步性能对比', 'Position', [100, 100, 1600, 900]);

    % 对每种测试条件进行测试
    for mod_type = modulation_types
        current_mod = mod_type{1};
        fprintf(fid, '=== 调制方式: %s ===\n\n', current_mod);

        for f_offset = freq_offsets
            for snr = snrs
                % 生成测试信号
                N = floor(fs * signal_length);  % 确保采样点数量正确
                t = (0:N-1)/fs;  % 生成对应的时间向量
                switch current_mod
                    case 'AM'
                        modulated_signal = cos(2*pi*(f_carrier + f_offset)*t);
                    case 'BPSK'
                        data = randi([0 1], 1, N);
                        bpsk_signal = 2*data - 1;
                        modulated_signal = bpsk_signal .* cos(2*pi*(f_carrier + f_offset)*t);
                    case 'QPSK'
                        data = randi([0 3], 1, N);
                        qpsk_signal = exp(1j * pi/2 * data);
                        modulated_signal = real(qpsk_signal .* exp(1j*2*pi*(f_carrier + f_offset)*t));
                    case '16-QAM'
                        data = randi([0 15], 1, N);
                        qam_signal = qammod(data, 16, 'UnitAveragePower', true);
                        modulated_signal = real(qam_signal .* exp(1j*2*pi*(f_carrier + f_offset)*t));
                    otherwise
                        error('未知的调制方式: %s', current_mod);
                end
                noisy_signal = awgn(modulated_signal, snr, 'measured');

                % 测试各种方法
                % 1. 平方律法
                tic;
                [freq_error1, snr_est1] = square_law_sync(noisy_signal, fs, f_carrier);
                time1 = toc;

                % 2. 原始Costas环法（使用优化后的参数）
                tic;
                [freq_error2, snr_est2] = costas_loop_sync(noisy_signal, fs, f_carrier, ...
                    best_params.noise_bw, best_params.damping, best_params.freq_max);
                time2 = toc;

                % 3. 改进Costas环法（使用优化后的参数）
                tic;
                [freq_error3, snr_est3, debug3] = improved_costas_sync(noisy_signal, fs, f_carrier, ...
                    best_params.noise_bw, best_params.damping, best_params.freq_max);
                time3 = toc;

                % 4. 多级同步法（使用优化后的参数）
                tic;
                [freq_error4, snr_est4, debug4] = multi_stage_costas_sync(noisy_signal, fs, f_carrier, ...
                    best_params.noise_bw, best_params.damping, best_params.freq_max);
                time4 = toc;

                % 5. PLL同步法
                tic;
                [freq_error5, snr_est5, debug5] = pll_sync(noisy_signal, fs, f_carrier, 1, 0.707, best_params.freq_max);
                time5 = toc;

                % 6. 粒子滤波器同步法
                tic;
                [freq_error6, snr_est6, debug6] = particle_filter_sync(noisy_signal, fs, f_carrier, 100, best_params.freq_max);
                time6 = toc;

                % 记录结果
                fprintf(fid, '测试条件:\n');
                fprintf(fid, '调制方式: %s\n', current_mod);
                fprintf(fid, '频率偏差: %.1f Hz\n', f_offset);
                fprintf(fid, 'SNR: %.0f dB\n\n', snr);

                % 记录各方法结果
                methods = {'square_law', 'original_costas', 'improved_costas', 'multi_stage', 'pll', 'particle_filter'};
                freq_errors = [freq_error1, freq_error2, freq_error3, freq_error4, freq_error5, freq_error6];
                snr_ests = [snr_est1, snr_est2, snr_est3, snr_est4, snr_est5, snr_est6];
                times = [time1, time2, time3, time4, time5, time6];

                for i = 1:length(methods)
                    if f_offset ~= 0
                        freq_err_percent = abs((freq_errors(i)-f_offset)/f_offset) * 100;
                    else
                        freq_err_percent = 0;
                    end
                    fprintf(fid, '%s结果:\n', display_names{i});
                    fprintf(fid, '估计频率误差: %.2f Hz\n', freq_errors(i));
                    fprintf(fid, '估计SNR: %.2f dB\n', snr_ests(i));
                    fprintf(fid, '频率误差精度: %.2f%%\n', freq_err_percent);
                    fprintf(fid, '处理时间: %.3f 秒\n\n', times(i));

                    % 更新统计信息
                    stats = update_statistics(stats, methods{i}, display_names{i}, f_offset, snr, ...
                        freq_err_percent, abs(snr_ests(i)-snr), times(i));
                end

                fprintf(fid, '-------------------\n\n');

                % 绘制同步过程（仅绘制多级同步器）
                plot_sync_process(debug4, f_offset, snr, t, current_mod);
            end
        end
    end

    % 初始化统计结构
    function stats = initialize_statistics()
        stats = struct();

        % 使用英文名称作为字段名称
        methods = {'square_law', 'original_costas', 'improved_costas', 'multi_stage', 'pll', 'particle_filter'};

        % 对应的中文显示名称
        display_names = {'平方变换法', '原始Costas环法', '改进Costas环法', '多级同步法', 'PLL同步法', '粒子滤波器同步法'};

        for i = 1:length(methods)
            stats.(methods{i}).freq_errors = [];
            stats.(methods{i}).snr_errors = [];
            stats.(methods{i}).times = [];
        end

        % 将显示名称存储在结构体中
        stats.method_display_names = display_names;
    end

    % 更新统计信息
    function stats = update_statistics(stats, method, display_name, f_offset, snr, ...
        freq_error_percent, snr_error, time)
        stats.(method).freq_errors(end+1, :) = [f_offset, snr, freq_error_percent];
        stats.(method).snr_errors(end+1, :) = [f_offset, snr, snr_error];
        stats.(method).times(end+1, :) = [f_offset, snr, time];
    end

    % 输出统计摘要
    function print_statistics_summary(fid, stats, modulation_types)
        fprintf(fid, '\n性能统计摘要\n');
        fprintf(fid, '===================\n\n');

        methods = {'square_law', 'original_costas', 'improved_costas', 'multi_stage', 'pll', 'particle_filter'};
        display_names = stats.method_display_names;

        for i = 1:length(methods)
            method = methods{i};
            display_name = display_names{i};
            fprintf(fid, '%s:\n', display_name);

            % 频率误差统计
            freq_errs = stats.(method).freq_errors(:,3);
            fprintf(fid, '频率误差统计:\n');
            fprintf(fid, '  平均值: %.2f%%\n', mean(freq_errs));
            fprintf(fid, '  中位数: %.2f%%\n', median(freq_errs));
            fprintf(fid, '  最大值: %.2f%%\n', max(freq_errs));
            fprintf(fid, '  最小值: %.2f%%\n', min(freq_errs));
            fprintf(fid, '  标准差: %.2f%%\n\n', std(freq_errs));

            % SNR误差统计
            snr_errs = stats.(method).snr_errors(:,3);
            fprintf(fid, 'SNR误差统计:\n');
            fprintf(fid, '  平均值: %.2f dB\n', mean(snr_errs));
            fprintf(fid, '  中位数: %.2f dB\n', median(snr_errs));
            fprintf(fid, '  最大值: %.2f dB\n', max(snr_errs));
            fprintf(fid, '  最小值: %.2f dB\n', min(snr_errs));
            fprintf(fid, '  标准差: %.2f dB\n\n', std(snr_errs));

            % 处理时间统计
            times = stats.(method).times(:,3);
            fprintf(fid, '处理时间统计:\n');
            fprintf(fid, '  平均值: %.3f 秒\n', mean(times));
            fprintf(fid, '  中位数: %.3f 秒\n', median(times));
            fprintf(fid, '  最大值: %.3f 秒\n', max(times));
            fprintf(fid, '  最小值: %.3f 秒\n\n', min(times));
        end
    end

    % 绘制同步过程
    function plot_sync_process(debug_info, f_offset, snr, t, mod_type)
        clf;

        % 频率估计过程
        subplot(3,2,1);
        plot(t, debug_info.tracking_stage.freq_history);
        hold on;
        plot([t(1), t(end)], [f_offset, f_offset], 'r--');
        title(sprintf('频率估计过程 (%s, offset=%.1fHz, SNR=%.0fdB)', mod_type, f_offset, snr));
        xlabel('时间 (s)');
        ylabel('频率 (Hz)');
        legend('估计值', '实际值');
        grid on;

        % 相位估计过程
        subplot(3,2,2);
        plot(t, debug_info.tracking_stage.phase_history);
        title('相位估计过程');
        xlabel('时间 (s)');
        ylabel('相位 (rad)');
        grid on;

        % 误差信号
        subplot(3,2,3);
        plot(t, debug_info.tracking_stage.error_signal);
        title('误差信号');
        xlabel('时间 (s)');
        ylabel('误差');
        grid on;

        % 粒子分布（仅适用于粒子滤波器同步法）
        subplot(3,2,4);
        if isfield(debug_info.tracking_stage, 'particles') && isfield(debug_info.tracking_stage, 'weights')
            scatter(debug_info.tracking_stage.particles, debug_info.tracking_stage.weights, 10, 'filled');
            title('粒子滤波器分布');
            xlabel('频率 (Hz)');
            ylabel('权重');
            grid on;
        else
            text(0.5, 0.5, '无粒子滤波器数据', 'HorizontalAlignment', 'center');
            axis off;
        end

        % 频率历史
        subplot(3,2,5);
        plot(t, debug_info.tracking_stage.freq_history);
        hold on;
        plot([t(1), t(end)], [f_offset, f_offset], 'r--');
        title('频率历史');
        xlabel('时间 (s)');
        ylabel('频率 (Hz)');
        grid on;

        % SNR历史
        if isfield(debug_info.tracking_stage, 'snr_estimate')
            subplot(3,2,6);
            plot(t, debug_info.tracking_stage.snr_estimate);
            title('SNR历史');
            xlabel('时间 (s)');
            ylabel('SNR (dB)');
            grid on;
        else
            subplot(3,2,6);
            text(0.5, 0.5, '无SNR历史数据', 'HorizontalAlignment', 'center');
            axis off;
        end

        drawnow;
        pause(0.1);
    end

    % 绘制性能对比图
    function plot_performance_comparison(stats, freq_offsets, modulation_types)
        % 创建性能对比图
        figure('Name', '性能对比', 'Position', [100, 100, 1600, 900]);

        methods = {'square_law', 'original_costas', 'improved_costas', 'multi_stage', 'pll', 'particle_filter'};
        display_names = stats.method_display_names;

        % 频率误差 vs 频率偏移
        subplot(3,2,1);
        hold on;
        for i = 1:length(methods)
            data = stats.(methods{i}).freq_errors;
            plot(data(:,1), data(:,3), 'o-', 'DisplayName', display_names{i});
        end
        xlabel('频率偏移 (Hz)');
        ylabel('频率误差 (%)');
        title('频率估计性能');
        grid on;
        legend('show', 'Location', 'best');

        % SNR误差 vs 频率偏移
        subplot(3,2,2);
        hold on;
        for i = 1:length(methods)
            data = stats.(methods{i}).snr_errors;
            plot(data(:,1), data(:,3), 'o-', 'DisplayName', display_names{i});
        end
        xlabel('频率偏移 (Hz)');
        ylabel('SNR误差 (dB)');
        title('SNR估计性能');
        grid on;
        legend('show', 'Location', 'best');

        % 处理时间 vs 频率偏移
        subplot(3,2,3);
        hold on;
        for i = 1:length(methods)
            data = stats.(methods{i}).times;
            plot(data(:,1), data(:,3), 'o-', 'DisplayName', display_names{i});
        end
        xlabel('频率偏移 (Hz)');
        ylabel('处理时间 (s)');
        title('计算复杂度');
        grid on;
        legend('show', 'Location', 'best');

        % 频率误差箱线图
        subplot(3,2,4);
        all_freq_errors = [];
        group_labels_freq = {};
        for i = 1:length(methods)
            current_data = stats.(methods{i}).freq_errors(:,3);
            all_freq_errors = [all_freq_errors; current_data(:)];
            group_labels_freq = [group_labels_freq; repmat(display_names(i), length(current_data), 1)];
        end
        boxplot(all_freq_errors, group_labels_freq, 'Labels', display_names);
        ylabel('频率误差 (%)');
        title('频率误差分布');
        grid on;

        % SNR误差箱线图
        subplot(3,2,5);
        all_snr_errors = [];
        group_labels_snr = {};
        for i = 1:length(methods)
            current_data = stats.(methods{i}).snr_errors(:,3);
            all_snr_errors = [all_snr_errors; current_data(:)];
            group_labels_snr = [group_labels_snr; repmat(display_names(i), length(current_data), 1)];
        end
        boxplot(all_snr_errors, group_labels_snr, 'Labels', display_names);
        ylabel('SNR误差 (dB)');
        title('SNR误差分布');
        grid on;

        % 处理时间箱线图
        subplot(3,2,6);
        all_time_errors = [];
        group_labels_time = {};
        for i = 1:length(methods)
            current_data = stats.(methods{i}).times(:,3);
            all_time_errors = [all_time_errors; current_data(:)];
            group_labels_time = [group_labels_time; repmat(display_names(i), length(current_data), 1)];
        end
        boxplot(all_time_errors, group_labels_time, 'Labels', display_names);
        ylabel('处理时间 (s)');
        title('处理时间分布');
        grid on;

        drawnow;
    end
end
```


## 测试结果

> 注意：测试结果文件不存在

### 性能测试图表

> 注意：性能图表目录不存在

