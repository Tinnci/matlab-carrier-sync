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
