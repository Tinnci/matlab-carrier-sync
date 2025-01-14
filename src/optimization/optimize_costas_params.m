% optimize_costas_params.m
%{
优化Costas环参数

问题描述：
1. 当前Costas环存在以下问题：
   - 小频偏（0.5~2 Hz）时性能很好
   - 大频偏（5~150 Hz）时性能不稳定
   - SNR估计普遍偏低

2. 参数选择的困境：
   - noise_bw（噪声带宽）影响跟踪速度和稳定性
   - damping（阻尼系数）影响系统响应特性
   - freq_max（最大频率限制）影响捕获范围
   这些参数之间存在相互制约关系，难以手动选择最优值

3. 优化目标：
   - 提高大频偏时的捕获性能
   - 保持小频偏时的高精度
   - 改善SNR估计准确度

优化方法：
1. 使用并行网格搜索遍历参数空间
2. 对每组参数进行Monte Carlo测试
3. 综合评估频率误差和SNR误差
4. 选择最优参数组合
%}

function best_params = optimize_costas_params()
    % 优化Costas环参数
    % 输出:
    %   best_params: 结构体，包含最优参数

    % 测试参数设置
    freq_offsets = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150];  % Hz 扩展到150Hz
    snrs = [10, 20, 30];  % dB
    fs = 1000;  % Hz
    f_carrier = 100;  % Hz
    signal_length = 10;  % 秒
    monte_carlo_runs = 10;  % 每组参数测试次数

    % 参数搜索范围
    noise_bw_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3];  % 扩展到0.3
    damping_range = [0.4, 0.5, 0.6, 0.707, 1.0];  % 包含更低的阻尼系数
    freq_max_range = [10, 25, 50, 100, 150];  % 扩展到150Hz

    % 使用 ndgrid 生成所有组合
    [A, B, C] = ndgrid(noise_bw_range, damping_range, freq_max_range);
    param_combinations = [A(:), B(:), C(:)];
    num_combinations = size(param_combinations, 1);

    % 初始化评分数组
    scores = zeros(num_combinations, 1);

    % 创建结果文件
    fid = fopen('optimization_results.txt', 'w');
    fprintf(fid, 'Costas环参数优化结果\n');
    fprintf(fid, '===================\n\n');

    % 开启并行池
    pool = gcp('nocreate');
    if isempty(pool)
        parpool;
    end

    % 并行参数搜索
    parfor idx = 1:num_combinations
        noise_bw = param_combinations(idx, 1);
        damping = param_combinations(idx, 2);
        freq_max = param_combinations(idx, 3);

        % 当前参数组合的性能统计
        freq_errors = [];
        snr_errors = [];

        % 对每种测试条件进行多次Monte Carlo测试
        for f_offset = freq_offsets
            for snr = snrs
                % 生成预先的信号模板
                t = 0:1/fs:signal_length;
                modulated_signal = cos(2*pi*(f_carrier + f_offset)*t);
                % 使用相同的随机种子确保信号一致性
                rng(0);  
                noisy_signal = awgn(modulated_signal, snr, 'measured');

                % 使用当前参数进行同步测试
                [freq_error, snr_est] = test_costas_params(noisy_signal, fs, f_carrier, ...
                    noise_bw, damping, freq_max);

                % 计算误差
                freq_err_percent = abs((freq_error - f_offset)/f_offset) * 100;
                snr_err_db = abs(snr_est - snr);

                freq_errors = [freq_errors, freq_err_percent];
                snr_errors = [snr_errors, snr_err_db];
            end
        end

        % 计算性能得分
        % 频率误差权重更大，因为这是主要优化目标
        freq_score = -mean(freq_errors);  % 负值因为误差越小越好
        snr_score = -mean(snr_errors);
        total_score = freq_score * 0.8 + snr_score * 0.2;

        % 存储得分
        scores(idx) = total_score;
    end

    % 将得分添加到 param_combinations
    param_combinations(:,4) = scores;

    % 找到最优参数
    [~, best_idx] = max(param_combinations(:,4));
    best_params.noise_bw = param_combinations(best_idx, 1);
    best_params.damping = param_combinations(best_idx, 2);
    best_params.freq_max = param_combinations(best_idx, 3);
    best_params.score = param_combinations(best_idx, 4);

    % 记录所有参数组合的结果
    for idx = 1:num_combinations
        noise_bw = param_combinations(idx, 1);
        damping = param_combinations(idx, 2);
        freq_max = param_combinations(idx, 3);
        total_score = param_combinations(idx, 4);

        fprintf(fid, '参数组合:\n');
        fprintf(fid, 'noise_bw: %.3f\n', noise_bw);
        fprintf(fid, 'damping: %.3f\n', damping);
        fprintf(fid, 'freq_max: %.1f Hz\n', freq_max);
        fprintf(fid, '总得分: %.2f\n\n', total_score);
    end

    % 记录最优参数
    fprintf(fid, '\n最优参数组合:\n');
    fprintf(fid, 'noise_bw: %.3f\n', best_params.noise_bw);
    fprintf(fid, 'damping: %.3f\n', best_params.damping);
    fprintf(fid, 'freq_max: %.1f Hz\n', best_params.freq_max);
    fprintf(fid, '最终得分: %.2f\n', best_params.score);

    fclose(fid);
    fprintf('优化完成，结果已保存到 optimization_results.txt\n');
end

function [freq_error, snr_estimate] = test_costas_params(signal, fs, f_carrier, noise_bw, damping, freq_max)
    % 用于测试特定参数组合的Costas环性能

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

        % 相位检测器
        error(n) = sign(I_arm(n)) * Q_arm(n);

        % 环路滤波器
        freq = freq + K2 * error(n);
        freq = max(min(freq, freq_max_rad), freq_min_rad);

        % 相位更新
        phase = phase + freq + K1 * error(n);
        phase = mod(phase + pi, 2*pi) - pi;

        freq_history(n) = freq;
    end

    % 计算频率误差
    steady_state_start = floor(N * 0.7);
    avg_freq_radians = mean(freq_history(steady_state_start:end));
    freq_error = avg_freq_radians * fs / (2 * pi);

    % 计算SNR估计
    I_steady = I_arm(steady_state_start:end);
    Q_steady = Q_arm(steady_state_start:end);
    signal_power = mean(I_steady.^2);
    noise_power = mean(Q_steady.^2);
    snr_estimate = 10 * log10(signal_power / noise_power);
    snr_estimate = min(max(snr_estimate, 0), 40);
end
