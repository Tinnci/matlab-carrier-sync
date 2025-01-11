%{
改进的Costas环同步器

问题背景：
1. 原始Costas环存在的问题：
   - 大频偏（>5Hz）时容易失锁
   - SNR估计精度低
   - 频率估计不够平滑
   - 捕获过程不够稳定

2. 改进方案：
   a) 增加预同步阶段：
      - 使用FFT进行粗频率估计
      - 提供更好的初始频率估计值
   
   b) 自适应环路参数：
      - 捕获阶段：大带宽，快速跟踪
      - 跟踪阶段：小带宽，提高精度
   
   c) 优化相位检测器：
      - 使用arctan检测器提高精度
      - 增加判决反馈减少噪声影响
   
   d) 改进SNR估计：
      - 使用功率谱密度估计
      - 分离信号和噪声成分

3. 预期效果：
   - 扩大频率捕获范围
   - 提高频率估计精度
   - 改善SNR估计准确度
   - 增强系统稳定性
%}

function [freq_error, snr_estimate, debug_info] = improved_costas_sync(signal, fs, f_carrier)
    % 改进的Costas环载波同步
    % 输入参数:
    %   signal: 输入信号
    %   fs: 采样频率
    %   f_carrier: 载波频率
    % 输出参数:
    %   freq_error: 估计的频率误差 (Hz)
    %   snr_estimate: 估计的信噪比 (dB)
    %   debug_info: 调试信息结构体
    
    % 预同步阶段参数
    fft_size = 2^nextpow2(length(signal));
    freq_resolution = fs/fft_size;
    
    % Costas环参数（低SNR优化）
    damping = 0.707;  % 临界阻尼
    initial_noise_bw = 0.05;  % 降低初始带宽，减少噪声影响
    final_noise_bw = 0.01;    % 降低最终带宽，提高精度
    freq_max = 20;            % 增大频率范围
    
    % 预同步：FFT粗频率估计
    [initial_freq_error, initial_snr] = fft_presync(signal, fs, f_carrier, fft_size);
    
    % 根据SNR调整参数
    if initial_snr < 0
        % 低SNR条件下的参数调整
        damping = 0.8;        % 增加阻尼
        initial_noise_bw = initial_noise_bw * 0.5;  % 进一步降低带宽
        final_noise_bw = final_noise_bw * 0.5;
    end
    
    % 初始化变量
    N = length(signal);
    phase = 0;
    freq = 2*pi*initial_freq_error/fs;  % 使用FFT估计作为初始值
    freq_history = zeros(1, N);
    phase_history = zeros(1, N);
    noise_bw_history = zeros(1, N);
    
    % 初始化I/Q分支
    I_arm = zeros(1, N);
    Q_arm = zeros(1, N);
    error = zeros(1, N);
    
    % Costas环处理
    for n = 1:N
        % 自适应噪声带宽
        progress = n/N;
        current_noise_bw = adaptive_noise_bw(initial_noise_bw, final_noise_bw, progress);
        noise_bw_history(n) = current_noise_bw;
        
        % 计算环路滤波器系数
        [K1, K2] = calculate_loop_coefficients(current_noise_bw, damping);
        
        % 生成本地载波
        I_carrier = cos(2*pi*f_carrier*(n-1)/fs + phase);
        Q_carrier = -sin(2*pi*f_carrier*(n-1)/fs + phase);
        
        % I/Q解调
        I_arm(n) = signal(n) * I_carrier;
        Q_arm(n) = signal(n) * Q_carrier;
        
        % 改进的相位检测器（arctan）
        error(n) = improved_phase_detector(I_arm(n), Q_arm(n));
        
        % 环路滤波器
        freq = freq + K2 * error(n);
        % 频率限幅
        freq = limit_frequency(freq, freq_max, fs);
        
        % 相位更新
        phase = phase + freq + K1 * error(n);
        phase = mod(phase + pi, 2*pi) - pi;
        
        % 记录历史
        freq_history(n) = freq;
        phase_history(n) = phase;
    end
    
    % 使用稳态数据计算最终结果
    [freq_error, snr_estimate] = calculate_final_estimates(freq_history, I_arm, Q_arm, fs);
    
    % 计算收敛时间
    conv_time = calculate_convergence_time(freq_history, fs, freq_error);
    
    % 收集调试信息
    debug_info = struct(...
        'freq_history', freq_history * fs/(2*pi), ...  % 转换为Hz
        'phase_history', phase_history, ...
        'noise_bw_history', noise_bw_history, ...
        'error_signal', error, ...
        'initial_freq_error', initial_freq_error, ...
        'initial_snr', initial_snr, ...
        'conv_time', conv_time);
end

function [freq_error, snr] = fft_presync(signal, fs, f_carrier, fft_size)
    % FFT预同步
    window = hanning(length(signal))';
    windowed_signal = signal .* window;
    
    % 计算FFT
    spectrum = fft(windowed_signal, fft_size);
    freq = (0:fft_size-1)*fs/fft_size;
    magnitude = abs(spectrum(1:fft_size/2));
    
    % 在载波频率附近搜索峰值
    search_range = 50;  % Hz
    carrier_bin = round(f_carrier*fft_size/fs) + 1;
    search_start = max(1, carrier_bin - round(search_range*fft_size/fs));
    search_end = min(length(magnitude), carrier_bin + round(search_range*fft_size/fs));
    
    % 确保搜索范围有效
    if search_start >= search_end || search_start < 1 || search_end > length(magnitude)
        % 如果搜索范围无效，使用整个频谱
        search_start = 1;
        search_end = length(magnitude);
    end
    
    [~, max_idx] = max(magnitude(search_start:search_end));
    max_idx = max_idx + search_start - 1;  % 调整为全局索引
    peak_freq = freq(max_idx);
    
    % 计算频率误差
    freq_error = peak_freq - f_carrier;
    
    % 估计SNR
    % 确保索引范围有效
    signal_range = max(1, max_idx-2):min(length(magnitude), max_idx+2);
    noise_range = [max(1, max_idx-50):max(1, max_idx-5), ...
                  min(length(magnitude), max_idx+5):min(length(magnitude), max_idx+50)];
    
    signal_power = mean(magnitude(signal_range).^2);
    noise_power = mean(magnitude(noise_range).^2);
    snr = 10*log10(signal_power/noise_power);
end

function noise_bw = adaptive_noise_bw(initial_bw, final_bw, progress)
    % 自适应调整噪声带宽
    % 使用sigmoid函数实现平滑过渡
    transition_point = 0.3;  % 30%处开始转换
    transition_width = 0.2;  % 转换区域宽度
    x = (progress - transition_point)/transition_width;
    sigmoid = 1/(1 + exp(5*x));  % sigmoid函数
    noise_bw = final_bw + (initial_bw - final_bw)*sigmoid;
end

function [K1, K2] = calculate_loop_coefficients(noise_bw, damping)
    % 计算环路滤波器系数
    K1 = 4 * damping * noise_bw / (1 + 2*damping*noise_bw + noise_bw^2);
    K2 = 4 * noise_bw^2 / (1 + 2*damping*noise_bw + noise_bw^2);
end

function error = improved_phase_detector(I, Q)
    % 改进的相位检测器
    % 使用arctan检测器，并加入判决反馈
    error = atan2(Q, I);
    
    % 加入软判决
    confidence = sqrt(I^2 + Q^2);
    error = error * tanh(confidence);  % 使用tanh限制误差幅度
end

function freq = limit_frequency(freq, freq_max, fs)
    % 频率限幅
    freq_max_rad = 2*pi*freq_max/fs;
    freq = max(min(freq, freq_max_rad), -freq_max_rad);
end

function [freq_error, snr_estimate] = calculate_final_estimates(freq_history, I_arm, Q_arm, fs)
    % 计算最终估计值
    N = length(freq_history);
    steady_state_start = floor(N * 0.7);  % 使用后30%的数据
    
    % 频率误差估计
    avg_freq_radians = mean(freq_history(steady_state_start:end));
    freq_error = avg_freq_radians * fs / (2*pi);
    
    % 改进的SNR估计
    I_steady = I_arm(steady_state_start:end);
    Q_steady = Q_arm(steady_state_start:end);
    
    % 使用功率谱密度估计
    [pxx, f] = pwelch(I_steady, [], [], [], fs);
    signal_band = abs(f) <= 5;  % 假设信号带宽为5Hz
    signal_power = mean(pxx(signal_band));
    noise_power = mean(pxx(~signal_band));
    
    snr_estimate = 10*log10(signal_power/noise_power);
    snr_estimate = min(max(snr_estimate, 0), 40);  % 限制范围
end

function conv_time = calculate_convergence_time(freq_history, fs, final_freq)
    % 计算收敛时间
    % 将频率历史转换为Hz
    freq_hz = freq_history * fs/(2*pi);
    
    % 定义收敛阈值（最终值的5%）
    threshold = abs(final_freq) * 0.05;
    
    % 找到首次进入阈值范围的时间点
    freq_error = abs(freq_hz - final_freq);
    conv_indices = find(freq_error <= threshold, 1);
    
    if isempty(conv_indices)
        conv_time = length(freq_history);  % 未收敛
    else
        conv_time = conv_indices;
    end
end 