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
│   │   ├───square_law_sync.m
├───test
│   ├───sync
│   │   ├───test_sync_methods.m
├───.gitignore
├───concat_matlab_files.ps1
├───main.m
├───optimization_results.txt
├───README.md
├───run_matlab_script.ps1
├───sync_results.txt
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
- src\sync\square_law_sync.m
- test\sync\test_sync_methods.m

---


## 源代码内容


### main.m

```matlab
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
    freq_offsets = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];  % Hz 扩展到100Hz
    snrs = [10, 20, 30];  % dB
    fs = 1000;  % Hz
    f_carrier = 100;  % Hz
    signal_length = 10;  % 秒
    monte_carlo_runs = 10;  % 每组参数测试次数
    
    % 参数搜索范围
    noise_bw_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3];  % 扩展到0.3
    damping_range = [0.4, 0.5, 0.6, 0.707, 1.0];  % 包含更低的阻尼系数
    freq_max_range = [10, 25, 50, 100, 150];  % 扩展到150Hz
    
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
    confidence = amplitude * tanh(amplitude);  % 使用tanh限制误差幅度
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

    freq = (-fft_size/2:fft_size/2-1)*(fs/fft_size);
    search_width_bins = round(search_range * fft_size / fs);
    carrier_bin = find(freq >= f_carrier, 1);
    if isempty(carrier_bin)
        error('载波频率超出FFT频谱范围。');
    end
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
    [~, max_idx] = max(avg_spectrum(search_indices));
    peak_freq = freq(search_indices(max_idx));

    % 使用抛物线插值提高频率分辨率
    if max_idx > 1 && max_idx < length(search_indices)
        alpha = avg_spectrum(search_indices(max_idx-1));
        beta = avg_spectrum(search_indices(max_idx));
        gamma = avg_spectrum(search_indices(max_idx+1));
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
% test_sync_methods.m
function test_sync_methods()
    % 测试参数
    fs = 1000;          % 采样频率
    f_carrier = 100;    % 载波频率
    signal_length = 10; % 信号长度（秒）

    % 扩展测试条件
    freq_offsets = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];  % 扩展到100Hz
    snrs = [10, 20, 30];  % dB

    % 创建结果文件
    fid = fopen('sync_results.txt', 'w');
    fprintf(fid, '载波同步系统测试结果（第二版）\n');
    fprintf(fid, '===================\n\n');

    % 运行参数优化并获取最优参数
    fprintf('开始参数优化...\n');
    best_params = optimize_costas_params();  % 确保您有此函数
    fprintf('参数优化完成。\n\n');

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
                if f_offset ~= 0
                    freq_err_percent = abs((freq_errors(i)-f_offset)/f_offset) * 100;
                else
                    freq_err_percent = 0;
                end
                fprintf(fid, '频率误差精度: %.2f%%\n', freq_err_percent);
                fprintf(fid, '处理时间: %.3f 秒\n\n', times(i));

                % 更新统计信息
                stats = update_statistics(stats, i, f_offset, snr, ...
                    freq_err_percent, abs(snr_ests(i)-snr), times(i));
            end

            fprintf(fid, '-------------------\n\n');

            % 绘制同步过程（仅绘制多级同步器）
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
    subplot(3,1,1);
    plot(t, debug_info.tracking_stage.freq_history);
    hold on;
    plot([t(1), t(end)], [f_offset, f_offset], 'r--');
    title(sprintf('频率估计过程 (offset=%.1fHz, SNR=%.0fdB)', f_offset, snr));
    xlabel('时间 (s)');
    ylabel('频率 (Hz)');
    legend('估计值', '实际值');
    grid on;

    % 相位估计过程
    subplot(3,1,2);
    plot(t, debug_info.tracking_stage.phase_history);
    title('相位估计过程');
    xlabel('时间 (s)');
    ylabel('相位 (rad)');
    grid on;

    % 误差信号
    subplot(3,1,3);
    plot(t, debug_info.tracking_stage.error_signal);
    title('误差信号');
    xlabel('时间 (s)');
    ylabel('误差');
    grid on;

    % 添加多级同步信息
    annotation('textbox', [0.15, 0.95, 0.7, 0.05], ...
        'String', sprintf('多级同步过程: 频率误差=%.2fHz, SNR=%.2fdB', ...
        debug_info.tracking_stage.freq_error, snr), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    drawnow;
    pause(0.1);
end

function plot_performance_comparison(stats, freq_offsets, methods)
    % 创建性能对比图
    figure('Name', '性能对比', 'Position', [100, 100, 1200, 800]);

    % 频率误差vs频率偏移
    subplot(2,2,1);
    hold on;
    for i = 1:length(methods)
        data = stats.freq_errors{i};
        for snr_val = unique(data(:,2))'
            mask = data(:,2) == snr_val;
            plot(data(mask,1), data(mask,3), 'o-', 'DisplayName', ...
                sprintf('%s (SNR=%ddB)', methods{i}, snr_val));
        end
    end
    xlabel('频率偏移 (Hz)');
    ylabel('频率误差 (%)');
    title('频率估计性能');
    grid on;
    legend('show', 'Location', 'best');

    % SNR误差vs频率偏移
    subplot(2,2,2);
    hold on;
    for i = 1:length(methods)
        data = stats.snr_errors{i};
        for snr_val = unique(data(:,2))'
            mask = data(:,2) == snr_val;
            plot(data(mask,1), data(mask,3), 'o-', 'DisplayName', ...
                sprintf('%s (SNR=%ddB)', methods{i}, snr_val));
        end
    end
    xlabel('频率偏移 (Hz)');
    ylabel('SNR误差 (dB)');
    title('SNR估计性能');
    grid on;
    legend('show', 'Location', 'best');

    % 处理时间vs频率偏移
    subplot(2,2,3);
    hold on;
    for i = 1:length(methods)
        data = stats.times{i};
        plot(data(:,1), data(:,3), 'o-', 'DisplayName', methods{i});
    end
    xlabel('频率偏移 (Hz)');
    ylabel('处理时间 (s)');
    title('计算复杂度');
    grid on;
    legend('show', 'Location', 'best');

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

### 同步测试结果

```
载波同步系统测试结果（第二版）
===================

测试条件:
频率偏差: 0.5 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 0.50 Hz
估计SNR: 24.32 dB
频率误差精度: 0.00%
处理时间: 0.002 秒

原始Costas环法结果:
估计频率误差: 0.52 Hz
估计SNR: 3.47 dB
频率误差精度: 3.72%
处理时间: 0.007 秒

改进Costas环法结果:
估计频率误差: 0.46 Hz
估计SNR: 0.00 dB
频率误差精度: 7.63%
处理时间: 0.075 秒

多级同步法结果:
估计频率误差: 0.46 Hz
估计SNR: 0.00 dB
频率误差精度: 7.63%
处理时间: 0.310 秒

-------------------

测试条件:
频率偏差: 0.5 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 0.50 Hz
估计SNR: 25.80 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.50 Hz
估计SNR: 4.02 dB
频率误差精度: 0.17%
处理时间: 0.003 秒

改进Costas环法结果:
估计频率误差: 0.43 Hz
估计SNR: 0.00 dB
频率误差精度: 13.28%
处理时间: 0.062 秒

多级同步法结果:
估计频率误差: 0.43 Hz
估计SNR: 0.00 dB
频率误差精度: 13.28%
处理时间: 0.282 秒

-------------------

测试条件:
频率偏差: 0.5 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 0.50 Hz
估计SNR: 25.98 dB
频率误差精度: 0.00%
处理时间: 0.002 秒

原始Costas环法结果:
估计频率误差: 0.51 Hz
估计SNR: 4.09 dB
频率误差精度: 1.07%
处理时间: 0.002 秒

改进Costas环法结果:
估计频率误差: 0.38 Hz
估计SNR: 0.00 dB
频率误差精度: 23.25%
处理时间: 0.066 秒

多级同步法结果:
估计频率误差: 0.38 Hz
估计SNR: 0.00 dB
频率误差精度: 23.25%
处理时间: 0.263 秒

-------------------

测试条件:
频率偏差: 1.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 1.00 Hz
估计SNR: 24.16 dB
频率误差精度: 0.00%
处理时间: 0.007 秒

原始Costas环法结果:
估计频率误差: 1.00 Hz
估计SNR: 3.40 dB
频率误差精度: 0.16%
处理时间: 0.002 秒

改进Costas环法结果:
估计频率误差: 0.97 Hz
估计SNR: 0.00 dB
频率误差精度: 3.39%
处理时间: 0.065 秒

多级同步法结果:
估计频率误差: 0.97 Hz
估计SNR: 0.00 dB
频率误差精度: 3.39%
处理时间: 0.276 秒

-------------------

测试条件:
频率偏差: 1.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 1.00 Hz
估计SNR: 25.81 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 1.00 Hz
估计SNR: 4.05 dB
频率误差精度: 0.45%
处理时间: 0.006 秒

改进Costas环法结果:
估计频率误差: 0.91 Hz
估计SNR: 0.00 dB
频率误差精度: 9.00%
处理时间: 0.060 秒

多级同步法结果:
估计频率误差: 0.91 Hz
估计SNR: 0.00 dB
频率误差精度: 9.00%
处理时间: 0.273 秒

-------------------

测试条件:
频率偏差: 1.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 1.00 Hz
估计SNR: 25.99 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 1.00 Hz
估计SNR: 4.03 dB
频率误差精度: 0.31%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 0.88 Hz
估计SNR: 0.00 dB
频率误差精度: 11.69%
处理时间: 0.055 秒

多级同步法结果:
估计频率误差: 0.88 Hz
估计SNR: 0.00 dB
频率误差精度: 11.69%
处理时间: 0.260 秒

-------------------

测试条件:
频率偏差: 2.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 2.00 Hz
估计SNR: 24.21 dB
频率误差精度: 0.00%
处理时间: 0.000 秒

原始Costas环法结果:
估计频率误差: 1.95 Hz
估计SNR: 3.42 dB
频率误差精度: 2.42%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 1.95 Hz
估计SNR: 0.00 dB
频率误差精度: 2.34%
处理时间: 0.058 秒

多级同步法结果:
估计频率误差: 1.95 Hz
估计SNR: 0.00 dB
频率误差精度: 2.34%
处理时间: 0.259 秒

-------------------

测试条件:
频率偏差: 2.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 2.00 Hz
估计SNR: 25.81 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 2.00 Hz
估计SNR: 4.00 dB
频率误差精度: 0.11%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 1.91 Hz
估计SNR: 0.00 dB
频率误差精度: 4.53%
处理时间: 0.058 秒

多级同步法结果:
估计频率误差: 1.91 Hz
估计SNR: 0.00 dB
频率误差精度: 4.53%
处理时间: 0.266 秒

-------------------

测试条件:
频率偏差: 2.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 2.00 Hz
估计SNR: 26.00 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 2.00 Hz
估计SNR: 4.13 dB
频率误差精度: 0.22%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 1.83 Hz
估计SNR: 0.00 dB
频率误差精度: 8.29%
处理时间: 0.063 秒

多级同步法结果:
估计频率误差: 1.83 Hz
估计SNR: 0.00 dB
频率误差精度: 8.29%
处理时间: 0.261 秒

-------------------

测试条件:
频率偏差: 5.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 5.00 Hz
估计SNR: 24.28 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 4.82 Hz
估计SNR: 3.39 dB
频率误差精度: 3.60%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 4.93 Hz
估计SNR: 0.00 dB
频率误差精度: 1.47%
处理时间: 0.063 秒

多级同步法结果:
估计频率误差: 4.93 Hz
估计SNR: 0.00 dB
频率误差精度: 1.47%
处理时间: 0.291 秒

-------------------

测试条件:
频率偏差: 5.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 5.00 Hz
估计SNR: 25.82 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 4.99 Hz
估计SNR: 4.09 dB
频率误差精度: 0.16%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 4.91 Hz
估计SNR: 0.00 dB
频率误差精度: 1.87%
处理时间: 0.050 秒

多级同步法结果:
估计频率误差: 4.91 Hz
估计SNR: 0.00 dB
频率误差精度: 1.87%
处理时间: 0.260 秒

-------------------

测试条件:
频率偏差: 5.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 5.00 Hz
估计SNR: 25.99 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 5.00 Hz
估计SNR: 4.17 dB
频率误差精度: 0.04%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 4.85 Hz
估计SNR: 0.00 dB
频率误差精度: 3.02%
处理时间: 0.056 秒

多级同步法结果:
估计频率误差: 4.85 Hz
估计SNR: 0.00 dB
频率误差精度: 3.02%
处理时间: 0.259 秒

-------------------

测试条件:
频率偏差: 10.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 10.00 Hz
估计SNR: 24.41 dB
频率误差精度: 0.00%
处理时间: 0.000 秒

原始Costas环法结果:
估计频率误差: 8.19 Hz
估计SNR: 3.13 dB
频率误差精度: 18.06%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 8.86 Hz
估计SNR: 0.00 dB
频率误差精度: 11.44%
处理时间: 0.063 秒

多级同步法结果:
估计频率误差: 8.86 Hz
估计SNR: 0.00 dB
频率误差精度: 11.44%
处理时间: 0.259 秒

-------------------

测试条件:
频率偏差: 10.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 10.00 Hz
估计SNR: 25.81 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 8.38 Hz
估计SNR: 3.82 dB
频率误差精度: 16.18%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 8.82 Hz
估计SNR: 0.00 dB
频率误差精度: 11.78%
处理时间: 0.057 秒

多级同步法结果:
估计频率误差: 8.82 Hz
估计SNR: 0.00 dB
频率误差精度: 11.78%
处理时间: 0.268 秒

-------------------

测试条件:
频率偏差: 10.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 10.00 Hz
估计SNR: 25.99 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 8.40 Hz
估计SNR: 3.91 dB
频率误差精度: 16.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 8.78 Hz
估计SNR: 0.00 dB
频率误差精度: 12.25%
处理时间: 0.059 秒

多级同步法结果:
估计频率误差: 8.78 Hz
估计SNR: 0.00 dB
频率误差精度: 12.25%
处理时间: 0.267 秒

-------------------

测试条件:
频率偏差: 20.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 20.00 Hz
估计SNR: 24.27 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -3.39 Hz
估计SNR: 0.00 dB
频率误差精度: 116.97%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 10.00 Hz
估计SNR: 0.00 dB
频率误差精度: 50.00%
处理时间: 0.058 秒

多级同步法结果:
估计频率误差: 10.00 Hz
估计SNR: 0.00 dB
频率误差精度: 50.00%
处理时间: 0.277 秒

-------------------

测试条件:
频率偏差: 20.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 20.00 Hz
估计SNR: 25.81 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -4.75 Hz
估计SNR: 0.00 dB
频率误差精度: 123.74%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 10.00 Hz
估计SNR: 0.00 dB
频率误差精度: 50.00%
处理时间: 0.056 秒

多级同步法结果:
估计频率误差: 10.00 Hz
估计SNR: 0.00 dB
频率误差精度: 50.00%
处理时间: 0.263 秒

-------------------

测试条件:
频率偏差: 20.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 20.00 Hz
估计SNR: 25.99 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -4.85 Hz
估计SNR: 0.00 dB
频率误差精度: 124.27%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 10.00 Hz
估计SNR: 0.00 dB
频率误差精度: 50.00%
处理时间: 0.059 秒

多级同步法结果:
估计频率误差: 10.00 Hz
估计SNR: 0.00 dB
频率误差精度: 50.00%
处理时间: 0.264 秒

-------------------

测试条件:
频率偏差: 30.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 30.00 Hz
估计SNR: 24.24 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -2.55 Hz
估计SNR: 0.00 dB
频率误差精度: 108.51%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 9.95 Hz
估计SNR: 0.00 dB
频率误差精度: 66.82%
处理时间: 0.057 秒

多级同步法结果:
估计频率误差: 9.95 Hz
估计SNR: 0.00 dB
频率误差精度: 66.82%
处理时间: 0.263 秒

-------------------

测试条件:
频率偏差: 30.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 30.00 Hz
估计SNR: 25.79 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -6.40 Hz
估计SNR: 0.00 dB
频率误差精度: 121.35%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 9.95 Hz
估计SNR: 0.00 dB
频率误差精度: 66.83%
处理时间: 0.062 秒

多级同步法结果:
估计频率误差: 9.95 Hz
估计SNR: 0.00 dB
频率误差精度: 66.83%
处理时间: 0.261 秒

-------------------

测试条件:
频率偏差: 30.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 30.00 Hz
估计SNR: 25.99 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 9.93 Hz
估计SNR: 0.00 dB
频率误差精度: 66.89%
处理时间: 0.060 秒

多级同步法结果:
估计频率误差: 9.93 Hz
估计SNR: 0.00 dB
频率误差精度: 66.89%
处理时间: 0.266 秒

-------------------

测试条件:
频率偏差: 40.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 40.00 Hz
估计SNR: 24.26 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -1.75 Hz
估计SNR: 0.00 dB
频率误差精度: 104.37%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 5.43 Hz
估计SNR: 0.00 dB
频率误差精度: 86.44%
处理时间: 0.059 秒

多级同步法结果:
估计频率误差: 5.43 Hz
估计SNR: 0.00 dB
频率误差精度: 86.44%
处理时间: 0.264 秒

-------------------

测试条件:
频率偏差: 40.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 40.00 Hz
估计SNR: 25.82 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.30 Hz
估计SNR: 0.00 dB
频率误差精度: 100.74%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 6.72 Hz
估计SNR: 0.00 dB
频率误差精度: 83.19%
处理时间: 0.063 秒

多级同步法结果:
估计频率误差: 6.72 Hz
估计SNR: 0.00 dB
频率误差精度: 83.19%
处理时间: 0.256 秒

-------------------

测试条件:
频率偏差: 40.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 40.00 Hz
估计SNR: 26.00 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 6.76 Hz
估计SNR: 0.00 dB
频率误差精度: 83.11%
处理时间: 0.060 秒

多级同步法结果:
估计频率误差: 6.76 Hz
估计SNR: 0.00 dB
频率误差精度: 83.11%
处理时间: 0.268 秒

-------------------

测试条件:
频率偏差: 50.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 50.00 Hz
估计SNR: 24.31 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -1.81 Hz
估计SNR: 0.00 dB
频率误差精度: 103.62%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 0.41 Hz
估计SNR: 0.00 dB
频率误差精度: 99.18%
处理时间: 0.057 秒

多级同步法结果:
估计频率误差: 0.41 Hz
估计SNR: 0.00 dB
频率误差精度: 99.18%
处理时间: 0.260 秒

-------------------

测试条件:
频率偏差: 50.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 50.00 Hz
估计SNR: 25.80 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 0.95 Hz
估计SNR: 0.00 dB
频率误差精度: 98.10%
处理时间: 0.067 秒

多级同步法结果:
估计频率误差: 0.95 Hz
估计SNR: 0.00 dB
频率误差精度: 98.10%
处理时间: 0.281 秒

-------------------

测试条件:
频率偏差: 50.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 50.00 Hz
估计SNR: 25.99 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: 0.26 Hz
估计SNR: 0.00 dB
频率误差精度: 99.49%
处理时间: 0.058 秒

多级同步法结果:
估计频率误差: 0.26 Hz
估计SNR: 0.00 dB
频率误差精度: 99.49%
处理时间: 0.263 秒

-------------------

测试条件:
频率偏差: 60.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: 60.00 Hz
估计SNR: 24.28 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -1.15 Hz
估计SNR: 0.00 dB
频率误差精度: 101.92%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -1.62 Hz
估计SNR: 0.00 dB
频率误差精度: 102.71%
处理时间: 0.056 秒

多级同步法结果:
估计频率误差: -1.62 Hz
估计SNR: 0.00 dB
频率误差精度: 102.71%
处理时间: 0.259 秒

-------------------

测试条件:
频率偏差: 60.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 60.00 Hz
估计SNR: 25.80 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -1.43 Hz
估计SNR: 0.00 dB
频率误差精度: 102.39%
处理时间: 0.060 秒

多级同步法结果:
估计频率误差: -1.43 Hz
估计SNR: 0.00 dB
频率误差精度: 102.39%
处理时间: 0.263 秒

-------------------

测试条件:
频率偏差: 60.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 60.00 Hz
估计SNR: 25.99 dB
频率误差精度: 0.00%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -0.97 Hz
估计SNR: 0.00 dB
频率误差精度: 101.62%
处理时间: 0.059 秒

多级同步法结果:
估计频率误差: -0.97 Hz
估计SNR: 0.00 dB
频率误差精度: 101.62%
处理时间: 0.264 秒

-------------------

测试条件:
频率偏差: 70.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: -49.15 Hz
估计SNR: -3.60 dB
频率误差精度: 170.21%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -1.36 Hz
估计SNR: 0.00 dB
频率误差精度: 101.94%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -2.24 Hz
估计SNR: 1.51 dB
频率误差精度: 103.20%
处理时间: 0.057 秒

多级同步法结果:
估计频率误差: -2.24 Hz
估计SNR: 1.51 dB
频率误差精度: 103.20%
处理时间: 0.262 秒

-------------------

测试条件:
频率偏差: 70.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 57.80 Hz
估计SNR: -10.12 dB
频率误差精度: 17.43%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 99.99%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -3.05 Hz
估计SNR: 1.27 dB
频率误差精度: 104.36%
处理时间: 0.055 秒

多级同步法结果:
估计频率误差: -3.05 Hz
估计SNR: 1.27 dB
频率误差精度: 104.36%
处理时间: 0.269 秒

-------------------

测试条件:
频率偏差: 70.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: -21.80 Hz
估计SNR: -19.76 dB
频率误差精度: 131.14%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -3.47 Hz
估计SNR: 2.50 dB
频率误差精度: 104.96%
处理时间: 0.062 秒

多级同步法结果:
估计频率误差: -3.47 Hz
估计SNR: 2.50 dB
频率误差精度: 104.96%
处理时间: 0.263 秒

-------------------

测试条件:
频率偏差: 80.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: -40.40 Hz
估计SNR: -4.15 dB
频率误差精度: 150.50%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -2.50 Hz
估计SNR: 0.00 dB
频率误差精度: 103.12%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -2.16 Hz
估计SNR: 7.56 dB
频率误差精度: 102.70%
处理时间: 0.061 秒

多级同步法结果:
估计频率误差: -2.16 Hz
估计SNR: 7.56 dB
频率误差精度: 102.70%
处理时间: 0.255 秒

-------------------

测试条件:
频率偏差: 80.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: -27.35 Hz
估计SNR: -10.41 dB
频率误差精度: 134.19%
处理时间: 0.000 秒

原始Costas环法结果:
估计频率误差: -0.14 Hz
估计SNR: 0.00 dB
频率误差精度: 100.17%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -3.72 Hz
估计SNR: 8.62 dB
频率误差精度: 104.66%
处理时间: 0.056 秒

多级同步法结果:
估计频率误差: -3.72 Hz
估计SNR: 8.62 dB
频率误差精度: 104.66%
处理时间: 0.260 秒

-------------------

测试条件:
频率偏差: 80.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: -38.70 Hz
估计SNR: -21.42 dB
频率误差精度: 148.38%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -3.84 Hz
估计SNR: 8.70 dB
频率误差精度: 104.80%
处理时间: 0.062 秒

多级同步法结果:
估计频率误差: -3.84 Hz
估计SNR: 8.70 dB
频率误差精度: 104.80%
处理时间: 0.283 秒

-------------------

测试条件:
频率偏差: 90.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: -10.65 Hz
估计SNR: -2.58 dB
频率误差精度: 111.83%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -3.69 Hz
估计SNR: 0.00 dB
频率误差精度: 104.10%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -2.48 Hz
估计SNR: 10.90 dB
频率误差精度: 102.75%
处理时间: 0.060 秒

多级同步法结果:
估计频率误差: -2.48 Hz
估计SNR: 10.90 dB
频率误差精度: 102.75%
处理时间: 0.260 秒

-------------------

测试条件:
频率偏差: 90.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: 54.50 Hz
估计SNR: -12.38 dB
频率误差精度: 39.44%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -4.92 Hz
估计SNR: 0.00 dB
频率误差精度: 105.46%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -3.42 Hz
估计SNR: 11.80 dB
频率误差精度: 103.80%
处理时间: 0.055 秒

多级同步法结果:
估计频率误差: -3.42 Hz
估计SNR: 11.80 dB
频率误差精度: 103.80%
处理时间: 0.258 秒

-------------------

测试条件:
频率偏差: 90.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: -56.30 Hz
估计SNR: -22.49 dB
频率误差精度: 162.56%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -4.98 Hz
估计SNR: 0.00 dB
频率误差精度: 105.53%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -4.10 Hz
估计SNR: 12.04 dB
频率误差精度: 104.55%
处理时间: 0.066 秒

多级同步法结果:
估计频率误差: -4.10 Hz
估计SNR: 12.04 dB
频率误差精度: 104.55%
处理时间: 0.262 秒

-------------------

测试条件:
频率偏差: 100.0 Hz
SNR: 10 dB

平方变换法结果:
估计频率误差: -10.10 Hz
估计SNR: -2.23 dB
频率误差精度: 110.10%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: -0.47 Hz
估计SNR: 0.00 dB
频率误差精度: 100.47%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -2.53 Hz
估计SNR: 11.48 dB
频率误差精度: 102.53%
处理时间: 0.058 秒

多级同步法结果:
估计频率误差: -2.53 Hz
估计SNR: 11.48 dB
频率误差精度: 102.53%
处理时间: 0.268 秒

-------------------

测试条件:
频率偏差: 100.0 Hz
SNR: 20 dB

平方变换法结果:
估计频率误差: -37.85 Hz
估计SNR: -13.01 dB
频率误差精度: 137.85%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -2.10 Hz
估计SNR: 12.25 dB
频率误差精度: 102.10%
处理时间: 0.057 秒

多级同步法结果:
估计频率误差: -2.10 Hz
估计SNR: 12.25 dB
频率误差精度: 102.10%
处理时间: 0.266 秒

-------------------

测试条件:
频率偏差: 100.0 Hz
SNR: 30 dB

平方变换法结果:
估计频率误差: 51.70 Hz
估计SNR: -22.03 dB
频率误差精度: 48.30%
处理时间: 0.001 秒

原始Costas环法结果:
估计频率误差: 0.00 Hz
估计SNR: 0.00 dB
频率误差精度: 100.00%
处理时间: 0.001 秒

改进Costas环法结果:
估计频率误差: -2.05 Hz
估计SNR: 12.33 dB
频率误差精度: 102.05%
处理时间: 0.059 秒

多级同步法结果:
估计频率误差: -2.05 Hz
估计SNR: 12.33 dB
频率误差精度: 102.05%
处理时间: 0.279 秒

-------------------


性能统计摘要
===================

平方变换法:
频率误差统计:
  平均值: 32.43%
  中位数: 0.00%
  最大值: 170.21%
  最小值: 0.00%
  标准差: 58.21%

SNR误差统计:
  平均值: 14.88 dB
  中位数: 12.40 dB
  最大值: 52.49 dB
  最小值: 4.00 dB
  标准差: 14.38 dB

处理时间统计:
  平均值: 0.001 秒
  中位数: 0.001 秒
  最大值: 0.007 秒
  最小值: 0.000 秒

原始Costas环法:
频率误差统计:
  平均值: 68.78%
  中位数: 100.00%
  最大值: 124.27%
  最小值: 0.04%
  标准差: 49.27%

SNR误差统计:
  平均值: 18.64 dB
  中位数: 20.00 dB
  最大值: 30.00 dB
  最小值: 6.53 dB
  标准差: 8.37 dB

处理时间统计:
  平均值: 0.001 秒
  中位数: 0.001 秒
  最大值: 0.007 秒
  最小值: 0.001 秒

改进Costas环法:
频率误差统计:
  平均值: 61.30%
  中位数: 75.00%
  最大值: 104.96%
  最小值: 1.47%
  标准差: 42.86%

SNR误差统计:
  平均值: 17.71 dB
  中位数: 20.00 dB
  最大值: 30.00 dB
  最小值: 0.90 dB
  标准差: 9.05 dB

处理时间统计:
  平均值: 0.060 秒
  中位数: 0.059 秒
  最大值: 0.075 秒
  最小值: 0.050 秒

多级同步法:
频率误差统计:
  平均值: 61.30%
  中位数: 75.00%
  最大值: 104.96%
  最小值: 1.47%
  标准差: 42.86%

SNR误差统计:
  平均值: 17.71 dB
  中位数: 20.00 dB
  最大值: 30.00 dB
  最小值: 0.90 dB
  标准差: 9.05 dB

处理时间统计:
  平均值: 0.267 秒
  中位数: 0.263 秒
  最大值: 0.310 秒
  最小值: 0.255 秒

```

### 性能测试图表

> 注意：性能图表目录不存在

