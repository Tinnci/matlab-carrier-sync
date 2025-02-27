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
