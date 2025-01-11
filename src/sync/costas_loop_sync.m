function [freq_error, snr_estimate, debug_info] = costas_loop_sync(signal, fs, f_carrier)
    % Costas环法实现载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率
    %   f_carrier: 载波频率
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)
    %   debug_info: 调试信息结构体
    
    % Costas环参数
    damping = 0.707;   % 临界阻尼
    noise_bw = 0.02;   % 降低噪声带宽，提高稳定性
    
    % 频率限幅设置
    freq_max = 2*pi*50/fs;  % 最大允许频率偏差±50Hz
    freq_min = -freq_max;
    
    % 初始化变量
    N = length(signal);
    phase = 0;
    freq = 0;
    freq_history = zeros(1, N);
    phase_history = zeros(1, N);
    noise_bw_history = zeros(1, N);
    
    % 环路滤波器系数计算
    K1 = 4 * damping * noise_bw / (1 + 2 * damping * noise_bw + noise_bw^2);
    K2 = 4 * noise_bw^2 / (1 + 2 * damping * noise_bw + noise_bw^2);
    
    % Costas环处理
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
        
        % 改进的相位检测器：使用符号判决来增强抗噪性能
        error(n) = sign(I_arm(n)) * Q_arm(n);
        
        % 环路滤波器
        freq = freq + K2 * error(n);
        % 频率限幅
        freq = max(min(freq, freq_max), freq_min);
        
        phase = phase + freq + K1 * error(n);
        % 相位归一化到[-pi, pi]
        phase = mod(phase + pi, 2*pi) - pi;
        
        % 记录历史
        freq_history(n) = freq * fs / (2*pi);  % 转换为Hz
        phase_history(n) = phase;
        noise_bw_history(n) = noise_bw;
    end
    
    % 计算频率误差（使用稳态频率）
    steady_state_start = floor(N * 0.7);  % 使用后30%的数据
    avg_freq_radians = mean(freq_history(steady_state_start:end));
    
    % 转换为Hz
    freq_error = avg_freq_radians;
    
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
    
    % 计算收敛时间
    conv_time = calculate_convergence_time(freq_history, freq_error);
    
    % 构建调试信息结构体
    debug_info = struct(...
        'freq_history', freq_history, ...
        'phase_history', phase_history, ...
        'noise_bw_history', noise_bw_history, ...
        'error_signal', error, ...
        'initial_freq_error', freq_history(1), ...
        'initial_snr', snr_estimate, ...
        'conv_time', conv_time);
end

function conv_time = calculate_convergence_time(freq_history, final_freq)
    % 计算收敛时间
    % 定义收敛阈值（最终值的5%）
    threshold = abs(final_freq) * 0.05;
    
    % 找到首次进入阈值范围的时间点
    freq_error = abs(freq_history - final_freq);
    conv_indices = find(freq_error <= threshold, 1);
    
    if isempty(conv_indices)
        conv_time = length(freq_history);  % 未收敛
    else
        conv_time = conv_indices;
    end
end 