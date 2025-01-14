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
