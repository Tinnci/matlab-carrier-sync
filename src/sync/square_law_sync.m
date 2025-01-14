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
