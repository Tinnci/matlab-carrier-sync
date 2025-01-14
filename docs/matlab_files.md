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
├───optimization_results.csv
├───README.md
├───run_matlab_script.ps1
├───sync_results.csv
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
% 优化Costas环参数
% 输出:
%   best_params: 结构体，包含最优参数

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

    % 调制方式
    modulation_types = {'BPSK', 'QPSK'};

    % 使用 ndgrid 生成所有组合
    [A, B, C] = ndgrid(noise_bw_range, damping_range, freq_max_range);
    param_combinations = [A(:), B(:), C(:)];
    num_combinations = size(param_combinations, 1);

    % 初始化评分数组
    scores = zeros(num_combinations, 1);

    % 开启并行池
    pool = gcp('nocreate');
    if isempty(pool)
        parpool;
    end

    % 初始化临时存储数组
    temp_scores = zeros(num_combinations, 1);

    % 并行参数搜索
    parfor idx = 1:num_combinations
        noise_bw = param_combinations(idx, 1);
        damping = param_combinations(idx, 2);
        freq_max = param_combinations(idx, 3);

        % 当前参数组合的性能统计
        total_tests = length(modulation_types) * length(freq_offsets) * length(snrs) * monte_carlo_runs;
        freq_errors = zeros(1, total_tests);
        snr_errors = zeros(1, total_tests);
        error_idx = 1;

        % 对每种调制方式进行测试
        for mod_idx = 1:length(modulation_types)
            modulation_type = modulation_types{mod_idx};

            % 对每种测试条件进行多次Monte Carlo测试
            for f_offset = freq_offsets
                for snr = snrs
                    for run = 1:monte_carlo_runs
                        % 生成预先的信号模板
                        t = 0:1/fs:signal_length;
                        % 初始化 modulated_signal 以避免未初始化警告
                        modulated_signal = zeros(1, length(t));

                        switch modulation_type
                            case 'BPSK'
                                data = randi([0 1], 1, length(t));
                                bpsk_signal = 2*data - 1;
                                modulated_signal = bpsk_signal .* cos(2*pi*(f_carrier + f_offset)*t);
                            case 'QPSK'
                                data_I = 2*randi([0 1],1,length(t)) -1;  % I分量：-1 或 +1
                                data_Q = 2*randi([0 1],1,length(t)) -1;  % Q分量：-1 或 +1
                                modulated_signal = data_I .* cos(2*pi*(f_carrier + f_offset)*t) + ...
                                                   data_Q .* sin(2*pi*(f_carrier + f_offset)*t);
                            otherwise
                                error('Unsupported modulation type: %s', modulation_type);
                        end
                        noisy_signal = awgn(modulated_signal, snr, 'measured');

                        % 使用当前参数进行同步测试
                        [freq_error, snr_est] = test_costas_params(noisy_signal, fs, f_carrier, ...
                            noise_bw, damping, freq_max, modulation_type);

                        % 计算误差
                        if f_offset ~=0
                            freq_err_percent = abs((freq_error - f_offset)/f_offset) * 100;
                        else
                            freq_err_percent = 0;
                        end
                        snr_err_db = abs(snr_est - snr);

                        freq_errors(error_idx) = freq_err_percent;
                        snr_errors(error_idx) = snr_err_db;
                        error_idx = error_idx + 1;
                    end
                end
            end
        end

        % 计算性能得分
        % 频率误差权重更大，因为这是主要优化目标
        freq_score = -mean(freq_errors);  % 负值因为误差越小越好
        snr_score = -mean(snr_errors);
        total_score = freq_score * 0.8 + snr_score * 0.2;

        % 存储得分
        temp_scores(idx) = total_score;
    end

    % 赋值给主评分数组
    scores = temp_scores;

    % 将参数组合与评分合并
    optimization_results = [param_combinations, scores];

    % 找到最优参数
    [~, best_idx] = max(scores);
    best_params.noise_bw = param_combinations(best_idx, 1);
    best_params.damping = param_combinations(best_idx, 2);
    best_params.freq_max = param_combinations(best_idx, 3);
    best_params.score = scores(best_idx);

    % 写入结果到文件
    fid = fopen('optimization_results.csv', 'w');
    if fid == -1
        error('无法打开文件 optimization_results.csv 进行写入。');
    end

    % 写入表头
    fprintf(fid, 'noise_bw,damping,freq_max,score\n');

    % 写入所有参数组合及其得分
    for idx = 1:num_combinations
        fprintf(fid, '%.3f,%.3f,%.1f,%.2f\n', ...
            param_combinations(idx,1), param_combinations(idx,2), param_combinations(idx,3), scores(idx));
    end

    % 写入最优参数
    fprintf(fid, '\nBest Parameters:\n');
    fprintf(fid, 'noise_bw,%.3f\n', best_params.noise_bw);
    fprintf(fid, 'damping,%.3f\n', best_params.damping);
    fprintf(fid, 'freq_max,%.1f\n', best_params.freq_max);
    fprintf(fid, 'score,%.2f\n', best_params.score);

    fclose(fid);
    fprintf('优化完成，结果已保存到 optimization_results.csv\n');
end

function [freq_error, snr_estimate] = test_costas_params(signal, fs, f_carrier, noise_bw, damping, freq_max, modulation_type)
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

    % 使用同步后的频率进行解调以计算 SNR
    synchronized_freq = f_carrier + freq_error;
    t_sync = (steady_state_start:N)/fs;  % 修正为 (steady_state_start:N)/fs
    I_steady = signal(steady_state_start:end) .* cos(2*pi*synchronized_freq*t_sync);
    Q_steady = signal(steady_state_start:end) .* -sin(2*pi*synchronized_freq*t_sync);

    % 计算信号功率和噪声功率
    if strcmp(modulation_type, 'BPSK')
        % 对于 BPSK，Q 分支主要包含噪声
        signal_power = mean(I_steady.^2);
        noise_power = mean(Q_steady.^2);
    elseif strcmp(modulation_type, 'QPSK')
        % 对于 QPSK，I 和 Q 分支都包含信号
        signal_power = mean(I_steady.^2 + Q_steady.^2) / 2;
        % 估计噪声功率（使用均值残差）
        noise_power = mean((I_steady - mean(I_steady)).^2 + (Q_steady - mean(Q_steady)).^2) / 2;
    else
        error('Unsupported modulation type: %s', modulation_type);
    end

    % 计算SNR
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
function [freq_error, snr_estimate] = costas_loop_sync(signal, fs, f_carrier, noise_bw, damping, freq_max, modulation_type)
    % Costas环法实现载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率 (Hz)
    %   f_carrier: 载波频率 (Hz)
    %   noise_bw: 噪声带宽
    %   damping: 阻尼系数
    %   freq_max: 最大频率偏移 (Hz)
    %   modulation_type: 'BPSK' 或 'QPSK'
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

    % 计算频率误差
    steady_state_start = floor(N * 0.7);
    avg_freq_radians = mean(freq_history(steady_state_start:end));
    freq_error = avg_freq_radians * fs / (2 * pi);

    % 使用同步后的频率进行解调以计算 SNR
    synchronized_freq = f_carrier + freq_error;
    t_sync = (steady_state_start:N)/fs;  % 修正为 (steady_state_start:N)/fs
    I_steady = signal(steady_state_start:end) .* cos(2*pi*synchronized_freq*t_sync);
    Q_steady = signal(steady_state_start:end) .* -sin(2*pi*synchronized_freq*t_sync);

    % 计算信号功率和噪声功率
    if strcmp(modulation_type, 'BPSK')
        % 对于 BPSK，Q 分支主要包含噪声
        signal_power = mean(I_steady.^2);
        noise_power = mean(Q_steady.^2);
    elseif strcmp(modulation_type, 'QPSK')
        % 对于 QPSK，I 和 Q 分支都包含信号
        signal_power = mean(I_steady.^2 + Q_steady.^2) / 2;
        % 估计噪声功率（使用均值残差）
        noise_power = mean((I_steady - mean(I_steady)).^2 + (Q_steady - mean(Q_steady)).^2) / 2;
    else
        error('Unsupported modulation type: %s', modulation_type);
    end

    % 计算 SNR
    snr_estimate = 10 * log10(signal_power / noise_power);
    snr_estimate = min(max(snr_estimate, 0), 40);  % 限制范围
end
```


### src\sync\improved_costas_sync.m

```matlab
% improved_costas_sync.m
function [freq_error, snr_estimate, debug_info] = improved_costas_sync(signal, fs, f_carrier, noise_bw, damping, freq_max, modulation_type)
    % 改进的Costas环载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率 (Hz)
    %   f_carrier: 载波频率 (Hz)
    %   noise_bw: 噪声带宽
    %   damping: 阻尼系数
    %   freq_max: 最大频率偏移 (Hz)
    %   modulation_type: 'BPSK' 或 'QPSK'
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
    [freq_error, snr_estimate] = calculate_final_estimates_tracked(freq_history, I_arm, Q_arm, fs, f_carrier, modulation_type);

    % 收集调试信息，并添加 freq_error 字段
    debug_info = struct(...
        'freq_history', freq_history * fs / (2 * pi), ...
        'phase_history', phase_history, ...
        'error_signal', error, ...
        'freq_error', freq_error);
end

function error = improved_phase_detector(I, Q)
    % 改进的相位检测器
    error = atan2(Q, I);

    % 加入软判决
    amplitude = sqrt(I^2 + Q^2);
    confidence = amplitude .* tanh(amplitude);  % 使用tanh限制误差幅度
    error = error .* confidence;
end

function [freq_error, snr_estimate] = calculate_final_estimates_tracked(freq_history, I_arm, Q_arm, fs, f_carrier, modulation_type)
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
function [freq_error, snr_estimate, debug_info] = multi_stage_costas_sync(signal, fs, f_carrier, noise_bw, damping, freq_max, modulation_type)
    % 多级Costas环同步器
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率 (Hz)
    %   f_carrier: 载波频率 (Hz)
    %   noise_bw: 噪声带宽 (优化后的)
    %   damping: 阻尼系数 (优化后的)
    %   freq_max: 最大频率偏移 (Hz) (优化后的)
    %   modulation_type: 'BPSK' 或 'QPSK'
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)
    %   debug_info: 调试信息结构体

    % 第一级：FFT粗搜索
    [coarse_freq_error, initial_snr] = wide_range_fft_search(signal, fs, f_carrier, 2^nextpow2(length(signal)), 200);

    % 第二级：分段精细搜索
    [refined_freq_error, refined_snr] = fine_grid_search(signal, fs, f_carrier, coarse_freq_error, 50);

    % 第三级：改进的Costas环精确跟踪（使用优化后的参数）
    [final_freq_error, final_snr, tracking_info] = improved_costas_sync(...
        signal, fs, f_carrier, noise_bw, damping, freq_max, modulation_type);

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
            if max(idx) > length(signal)
                break;
            end
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
function [freq_error, snr_estimate, debug_info] = particle_filter_sync(signal, fs, f_carrier, num_particles, freq_max, modulation_type)
    % 基于粒子滤波器的载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率 (Hz)
    %   f_carrier: 载波频率 (Hz)
    %   num_particles: 粒子数量
    %   freq_max: 最大频率偏移 (Hz)
    %   modulation_type: 'BPSK' 或 'QPSK'
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
        % 当前采样时间
        t = (n-1)/fs;
        
        % 生成本地载波
        I_carrier = cos(2 * pi * particles * t);
        Q_carrier = -sin(2 * pi * particles * t);

        % I/Q解调
        I_arm = signal(n) .* I_carrier;
        Q_arm = signal(n) .* Q_carrier;

        % 计算观测（相位）
        obs = atan2(Q_arm, I_arm);

        % 计算权重（假设相位噪声为高斯分布）
        sigma = 0.1;  % 相位噪声标准差
        weights = weights .* exp(- (obs).^2 / (2 * sigma^2));
        weights = weights / sum(weights);

        % 检查权重归一化
        if any(isnan(weights)) || any(weights < 0)
            weights = ones(1, num_particles) / num_particles;
        end

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

    % 使用同步后的频率进行解调以计算 SNR
    synchronized_freq = f_carrier + freq_error;
    t_sync = (steady_state_start:N)/fs;  % 修正为 (steady_state_start:N)/fs
    I_steady = signal(steady_state_start:end) .* cos(2*pi*synchronized_freq*t_sync);
    Q_steady = signal(steady_state_start:end) .* -sin(2*pi*synchronized_freq*t_sync);

    % 计算信号功率和噪声功率
    if strcmp(modulation_type, 'BPSK')
        % 对于 BPSK，Q 分支主要包含噪声
        signal_power = mean(I_steady.^2);
        noise_power = mean(Q_steady.^2);
    elseif strcmp(modulation_type, 'QPSK')
        % 对于 QPSK，I 和 Q 分支都包含信号
        signal_power = mean(I_steady.^2 + Q_steady.^2) / 2;
        % 估计噪声功率（使用均值残差）
        noise_power = mean((I_steady - mean(I_steady)).^2 + (Q_steady - mean(Q_steady)).^2) / 2;
    else
        error('Unsupported modulation type: %s', modulation_type);
    end

    % 计算 SNR
    snr_estimate = 10 * log10(signal_power / noise_power);
    snr_estimate = min(max(snr_estimate, 0), 40);  % 限制范围

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
function [freq_error, snr_estimate, debug_info] = pll_sync(signal, fs, f_carrier, loop_bw, damping, freq_max, modulation_type)
    % 基于锁相环（PLL）的载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率 (Hz)
    %   f_carrier: 载波频率 (Hz)
    %   loop_bw: 环路带宽
    %   damping: 阻尼系数
    %   freq_max: 最大频率偏移 (Hz)
    %   modulation_type: 'BPSK' 或 'QPSK'
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

    % 使用同步后的频率进行解调以计算 SNR
    synchronized_freq = f_carrier + freq_error;
    t_sync = (steady_state_start:N)/fs;  % 修正为 (steady_state_start:N)/fs
    I_steady = signal(steady_state_start:end) .* cos(2*pi*synchronized_freq*t_sync);
    Q_steady = signal(steady_state_start:end) .* -sin(2*pi*synchronized_freq*t_sync);

    % 计算信号功率和噪声功率
    if strcmp(modulation_type, 'BPSK')
        % 对于 BPSK，Q 分支主要包含噪声
        signal_power = mean(I_steady.^2);
        noise_power = mean(Q_steady.^2);
    elseif strcmp(modulation_type, 'QPSK')
        % 对于 QPSK，I 和 Q 分支都包含信号
        signal_power = mean(I_steady.^2 + Q_steady.^2) / 2;
        % 估计噪声功率（使用均值残差）
        noise_power = mean((I_steady - mean(I_steady)).^2 + (Q_steady - mean(Q_steady)).^2) / 2;
    else
        error('Unsupported modulation type: %s', modulation_type);
    end

    % 计算 SNR
    snr_estimate = 10 * log10(signal_power / noise_power);
    snr_estimate = min(max(snr_estimate, 0), 40);  % 限制范围

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
function [freq_error, snr_estimate] = square_law_sync(signal, fs, f_carrier, modulation_type)
    % 平方变换法实现载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率 (Hz)
    %   f_carrier: 载波频率 (Hz)
    %   modulation_type: 'BPSK' 或 'QPSK'
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)

    % 平方变换
    squared_signal = signal.^2;

    % FFT分析
    N = length(squared_signal);
    fft_size = 2^nextpow2(N);
    fft_result = fft(squared_signal, fft_size);
    fft_result = fftshift(fft_result);
    magnitude = abs(fft_result);
    freq = (-fft_size/2:fft_size/2-1) * (fs / fft_size);

    % 寻找二倍频成分的峰值
    expected_freq = 2 * f_carrier;  % 二倍载波频率
    [~, center_idx] = min(abs(freq - expected_freq));  % 找到接近二倍频的频谱点
    search_width = floor(fft_size / 16);  % 搜索范围宽度
    search_start = max(1, center_idx - search_width);
    search_end = min(length(freq), center_idx + search_width);
    search_range = search_start:search_end;

    [peak_value, local_peak_idx] = max(magnitude(search_range));
    peak_idx = search_start + local_peak_idx - 1;
    peak_freq = freq(peak_idx);

    % 计算频率误差（考虑二倍频的影响）
    freq_error = (peak_freq / 2) - f_carrier;

    % 使用同步后的频率进行解调以计算 SNR
    synchronized_freq = f_carrier + freq_error;
    t_sync = (0:length(signal)-1)/fs;
    I_sync = signal .* cos(2*pi*synchronized_freq*t_sync);
    Q_sync = signal .* -sin(2*pi*synchronized_freq*t_sync);

    % 计算信号功率和噪声功率
    if strcmp(modulation_type, 'BPSK')
        % 对于 BPSK，Q 分支主要包含噪声
        signal_power = mean(I_sync.^2);
        noise_power = mean(Q_sync.^2);
    elseif strcmp(modulation_type, 'QPSK')
        % 对于 QPSK，I 和 Q 分支都包含信号
        signal_power = mean(I_sync.^2 + Q_sync.^2) / 2;
        % 估计噪声功率（使用均值残差）
        noise_power = mean((I_sync - mean(I_sync)).^2 + (Q_sync - mean(Q_sync)).^2) / 2;
    else
        error('Unsupported modulation type: %s', modulation_type);
    end

    % 计算SNR
    snr_estimate = 10 * log10(signal_power / noise_power);
    snr_estimate = min(max(snr_estimate, 0), 40);  % 限制范围
end 
```


### test\sync\test_sync_methods.m

```matlab
% test_sync_methods.m
function test_sync_methods()
    % 测试不同同步方法的性能
    % 输出:
    %   sync_results.csv: 各同步方法在不同测试条件下的性能结果

    % 测试参数
    fs = 1000;          % 采样频率
    f_carrier = 100;    % 载波频率
    signal_length = 10; % 信号长度（秒）

    % 扩展测试条件
    freq_offsets = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150];  % Hz 扩展到150Hz
    snrs = [10, 20, 30];  % dB

    % 调制方式
    modulation_types = {'BPSK', 'QPSK'};

    % 创建结果文件并写入表头
    fid = fopen('sync_results.csv', 'w');
    if fid == -1
        error('无法创建或打开文件 sync_results.csv 进行写入。');
    end
    methods = {'square_law', 'original_costas', 'improved_costas', 'multi_stage', 'pll', 'particle_filter'};
    display_names = {'平方变换法', '原始Costas环法', '改进Costas环法', '多级同步法', 'PLL同步法', '粒子滤波器同步法'};
    header = ['调制方式,频率偏差 (Hz),SNR (dB)', ...
              ',平方变换法_频率误差 (Hz),平方变换法_SNR (dB),平方变换法_精度 (%)', ...
              ',原始Costas环法_频率误差 (Hz),原始Costas环法_SNR (dB),原始Costas环法_精度 (%)', ...
              ',改进Costas环法_频率误差 (Hz),改进Costas环法_SNR (dB),改进Costas环法_精度 (%)', ...
              ',多级同步法_频率误差 (Hz),多级同步法_SNR (dB),多级同步法_精度 (%)', ...
              ',PLL同步法_频率误差 (Hz),PLL同步法_SNR (dB),PLL同步法_精度 (%)', ...
              ',粒子滤波器同步法_频率误差 (Hz),粒子滤波器同步法_SNR (dB),粒子滤波器同步法_精度 (%),处理时间 (s)'];
    fprintf(fid, '%s\n', header);

    % 运行参数优化并获取最优参数
    fprintf('开始参数优化...\n');
    best_params = optimize_costas_params();  % 确保您有此函数
    fprintf('参数优化完成。\n\n');

    % 对每种测试条件进行测试
    for mod_type = modulation_types
        current_mod = mod_type{1};
        fprintf('测试调制方式: %s\n', current_mod);

        for f_offset = freq_offsets
            for snr = snrs
                % 生成测试信号
                N = floor(fs * signal_length);  % 采样点数量
                t = (0:N-1)/fs;  % 时间向量
                switch current_mod
                    case 'BPSK'
                        data = randi([0 1], 1, N);
                        bpsk_signal = 2*data - 1;
                        modulated_signal = bpsk_signal .* cos(2*pi*(f_carrier + f_offset)*t);
                    case 'QPSK'
                        % 正确生成QPSK信号：使用独立的I和Q数据流
                        data_I = 2*randi([0 1],1,N) -1;  % I分量：-1 或 +1
                        data_Q = 2*randi([0 1],1,N) -1;  % Q分量：-1 或 +1
                        modulated_signal = data_I .* cos(2*pi*(f_carrier + f_offset)*t) + ...
                                           data_Q .* sin(2*pi*(f_carrier + f_offset)*t);
                    otherwise
                        error('未知的调制方式: %s', current_mod);
                end
                noisy_signal = awgn(modulated_signal, snr, 'measured');

                % 测试各种方法
                % 1. 平方律法
                tic;
                [freq_error1, snr_est1] = square_law_sync(noisy_signal, fs, f_carrier, current_mod);
                time1 = toc;

                % 2. 原始Costas环法（使用优化后的参数）
                tic;
                [freq_error2, snr_est2] = costas_loop_sync(noisy_signal, fs, f_carrier, ...
                    best_params.noise_bw, best_params.damping, best_params.freq_max, current_mod);
                time2 = toc;

                % 3. 改进Costas环法（使用优化后的参数）
                tic;
                [freq_error3, snr_est3, ~] = improved_costas_sync(noisy_signal, fs, f_carrier, ...
                    best_params.noise_bw, best_params.damping, best_params.freq_max, current_mod);
                time3 = toc;

                % 4. 多级同步法（使用优化后的参数）
                tic;
                [freq_error4, snr_est4, ~] = multi_stage_costas_sync(noisy_signal, fs, f_carrier, ...
                    best_params.noise_bw, best_params.damping, best_params.freq_max, current_mod);
                time4 = toc;

                % 5. PLL同步法
                tic;
                [freq_error5, snr_est5, ~] = pll_sync(noisy_signal, fs, f_carrier, 1, 0.707, best_params.freq_max, current_mod);
                time5 = toc;

                % 6. 粒子滤波器同步法
                tic;
                [freq_error6, snr_est6, ~] = particle_filter_sync(noisy_signal, fs, f_carrier, 100, best_params.freq_max, current_mod);
                time6 = toc;

                % 计算频率误差精度
                if f_offset ~= 0
                    freq_err_percent1 = abs((freq_error1 - f_offset)/f_offset) * 100;
                    freq_err_percent2 = abs((freq_error2 - f_offset)/f_offset) * 100;
                    freq_err_percent3 = abs((freq_error3 - f_offset)/f_offset) * 100;
                    freq_err_percent4 = abs((freq_error4 - f_offset)/f_offset) * 100;
                    freq_err_percent5 = abs((freq_error5 - f_offset)/f_offset) * 100;
                    freq_err_percent6 = abs((freq_error6 - f_offset)/f_offset) * 100;
                else
                    freq_err_percent1 = 0;
                    freq_err_percent2 = 0;
                    freq_err_percent3 = 0;
                    freq_err_percent4 = 0;
                    freq_err_percent5 = 0;
                    freq_err_percent6 = 0;
                end

                % 格式化输出到CSV
                fprintf(fid, '%s,%.1f,%d', current_mod, f_offset, snr);
                fprintf(fid, ',%.2f,%.2f,%.2f', freq_error1, snr_est1, freq_err_percent1);
                fprintf(fid, ',%.2f,%.2f,%.2f', freq_error2, snr_est2, freq_err_percent2);
                fprintf(fid, ',%.2f,%.2f,%.2f', freq_error3, snr_est3, freq_err_percent3);
                fprintf(fid, ',%.2f,%.2f,%.2f', freq_error4, snr_est4, freq_err_percent4);
                fprintf(fid, ',%.2f,%.2f,%.2f', freq_error5, snr_est5, freq_err_percent5);
                fprintf(fid, ',%.2f,%.2f,%.2f,%.3f\n', freq_error6, snr_est6, freq_err_percent6, time6);
            end
        end
    end
```


## 测试结果

> 注意：测试结果文件不存在

### 性能测试图表

> 注意：性能图表目录不存在

