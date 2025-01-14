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
