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
