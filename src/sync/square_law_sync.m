function [freq_error, snr_estimate, debug_info] = square_law_sync(signal, fs, f_carrier)
    % 平方变换法实现载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率
    %   f_carrier: 载波频率
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)
    %   debug_info: 调试信息结构体
    
    % 初始化历史记录
    N = length(signal);
    freq_history = zeros(1, N);
    phase_history = zeros(1, N);
    error_signal = zeros(1, N);
    noise_bw_history = zeros(1, N);
    
    % 平方变换
    squared_signal = signal.^2;
    
    % FFT分析
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
    
    % 填充历史记录
    for n = 1:N
        freq_history(n) = freq_error;
        phase_history(n) = 2*pi*freq_error*n/fs;
        error_signal(n) = peak_value - magnitude(center_idx);
        noise_bw_history(n) = search_width*fs/N;
    end
    
    % 改进的SNR估计
    signal_range = max(1, peak_idx - 2):min(N, peak_idx + 2);
    signal_power = mean(magnitude(signal_range).^2);
    
    exclude_range = max(1, peak_idx - 10):min(N, peak_idx + 10);
    noise_magnitude = magnitude;
    noise_magnitude(exclude_range) = [];
    noise_power = mean(noise_magnitude.^2);
    
    snr_estimate = 10 * log10(signal_power / noise_power);
    
    % 构建调试信息结构体
    debug_info = struct(...
        'freq_history', freq_history, ...
        'phase_history', phase_history, ...
        'noise_bw_history', noise_bw_history, ...
        'error_signal', error_signal, ...
        'initial_freq_error', freq_error, ...
        'initial_snr', snr_estimate, ...
        'conv_time', 1);  % 平方律法为非迭代算法，收敛时间设为1
end 