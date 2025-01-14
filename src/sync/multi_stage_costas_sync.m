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
