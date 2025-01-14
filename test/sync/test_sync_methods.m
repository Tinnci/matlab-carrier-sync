% test_sync_methods.m
function test_sync_methods()
    % 测试不同同步方法的性能
    % 输出:
    %   sync_results.csv: 各同步方法在不同测试条件下的性能结果

    % 测试参数
    fs = 1000;          % 采样频率
    f_carrier = 100;    % 载波频率
    signal_length = 10; % 信号长度（秒）

    % 扩展测试条件
    freq_offsets = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150];  % Hz 扩展到150Hz
    snrs = [10, 20, 30];  % dB

    % 调制方式
    modulation_types = {'BPSK', 'QPSK'};

    % 创建结果文件并写入表头
    fid = fopen('sync_results.csv', 'w');
    if fid == -1
        error('无法创建或打开文件 sync_results.csv 进行写入。');
    end
    methods = {'square_law', 'original_costas', 'improved_costas', 'multi_stage', 'pll', 'particle_filter'};
    display_names = {'平方变换法', '原始Costas环法', '改进Costas环法', '多级同步法', 'PLL同步法', '粒子滤波器同步法'};
    header = ['调制方式,频率偏差 (Hz),SNR (dB)', ...
              ',平方变换法_频率误差 (Hz),平方变换法_SNR (dB),平方变换法_精度 (%)', ...
              ',原始Costas环法_频率误差 (Hz),原始Costas环法_SNR (dB),原始Costas环法_精度 (%)', ...
              ',改进Costas环法_频率误差 (Hz),改进Costas环法_SNR (dB),改进Costas环法_精度 (%)', ...
              ',多级同步法_频率误差 (Hz),多级同步法_SNR (dB),多级同步法_精度 (%)', ...
              ',PLL同步法_频率误差 (Hz),PLL同步法_SNR (dB),PLL同步法_精度 (%)', ...
              ',粒子滤波器同步法_频率误差 (Hz),粒子滤波器同步法_SNR (dB),粒子滤波器同步法_精度 (%),处理时间 (s)'];
    fprintf(fid, '%s\n', header);

    % 运行参数优化并获取最优参数
    fprintf('开始参数优化...\n');
    best_params = optimize_costas_params();  % 确保您有此函数
    fprintf('参数优化完成。\n\n');

    % 对每种测试条件进行测试
    for mod_type = modulation_types
        current_mod = mod_type{1};
        fprintf('测试调制方式: %s\n', current_mod);

        for f_offset = freq_offsets
            for snr = snrs
                % 生成测试信号
                N = floor(fs * signal_length);  % 采样点数量
                t = (0:N-1)/fs;  % 时间向量
                switch current_mod
                    case 'BPSK'
                        data = randi([0 1], 1, N);
                        bpsk_signal = 2*data - 1;
                        modulated_signal = bpsk_signal .* cos(2*pi*(f_carrier + f_offset)*t);
                    case 'QPSK'
                        % 正确生成QPSK信号：使用独立的I和Q数据流
                        data_I = 2*randi([0 1],1,N) -1;  % I分量：-1 或 +1
                        data_Q = 2*randi([0 1],1,N) -1;  % Q分量：-1 或 +1
                        modulated_signal = data_I .* cos(2*pi*(f_carrier + f_offset)*t) + ...
                                           data_Q .* sin(2*pi*(f_carrier + f_offset)*t);
                    otherwise
                        error('未知的调制方式: %s', current_mod);
                end
                noisy_signal = awgn(modulated_signal, snr, 'measured');

                % 测试各种方法
                % 1. 平方律法
                tic;
                [freq_error1, snr_est1] = square_law_sync(noisy_signal, fs, f_carrier, current_mod);
                time1 = toc;

                % 2. 原始Costas环法（使用优化后的参数）
                tic;
                [freq_error2, snr_est2] = costas_loop_sync(noisy_signal, fs, f_carrier, ...
                    best_params.noise_bw, best_params.damping, best_params.freq_max, current_mod);
                time2 = toc;

                % 3. 改进Costas环法（使用优化后的参数）
                tic;
                [freq_error3, snr_est3, ~] = improved_costas_sync(noisy_signal, fs, f_carrier, ...
                    best_params.noise_bw, best_params.damping, best_params.freq_max, current_mod);
                time3 = toc;

                % 4. 多级同步法（使用优化后的参数）
                tic;
                [freq_error4, snr_est4, ~] = multi_stage_costas_sync(noisy_signal, fs, f_carrier, ...
                    best_params.noise_bw, best_params.damping, best_params.freq_max, current_mod);
                time4 = toc;

                % 5. PLL同步法
                tic;
                [freq_error5, snr_est5, ~] = pll_sync(noisy_signal, fs, f_carrier, 1, 0.707, best_params.freq_max, current_mod);
                time5 = toc;

                % 6. 粒子滤波器同步法
                tic;
                [freq_error6, snr_est6, ~] = particle_filter_sync(noisy_signal, fs, f_carrier, 100, best_params.freq_max, current_mod);
                time6 = toc;

                % 计算频率误差精度
                if f_offset ~= 0
                    freq_err_percent1 = abs((freq_error1 - f_offset)/f_offset) * 100;
                    freq_err_percent2 = abs((freq_error2 - f_offset)/f_offset) * 100;
                    freq_err_percent3 = abs((freq_error3 - f_offset)/f_offset) * 100;
                    freq_err_percent4 = abs((freq_error4 - f_offset)/f_offset) * 100;
                    freq_err_percent5 = abs((freq_error5 - f_offset)/f_offset) * 100;
                    freq_err_percent6 = abs((freq_error6 - f_offset)/f_offset) * 100;
                else
                    freq_err_percent1 = 0;
                    freq_err_percent2 = 0;
                    freq_err_percent3 = 0;
                    freq_err_percent4 = 0;
                    freq_err_percent5 = 0;
                    freq_err_percent6 = 0;
                end

                % 格式化输出到CSV
                fprintf(fid, '%s,%.1f,%d', current_mod, f_offset, snr);
                fprintf(fid, ',%.2f,%.2f,%.2f', freq_error1, snr_est1, freq_err_percent1);
                fprintf(fid, ',%.2f,%.2f,%.2f', freq_error2, snr_est2, freq_err_percent2);
                fprintf(fid, ',%.2f,%.2f,%.2f', freq_error3, snr_est3, freq_err_percent3);
                fprintf(fid, ',%.2f,%.2f,%.2f', freq_error4, snr_est4, freq_err_percent4);
                fprintf(fid, ',%.2f,%.2f,%.2f', freq_error5, snr_est5, freq_err_percent5);
                fprintf(fid, ',%.2f,%.2f,%.2f,%.3f\n', freq_error6, snr_est6, freq_err_percent6, time6);
            end
        end
    end
