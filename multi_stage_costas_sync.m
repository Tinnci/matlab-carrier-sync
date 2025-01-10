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