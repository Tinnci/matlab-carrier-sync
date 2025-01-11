function results = test_high_freq_sync()
    % 高频载波同步测试函数
    % 测试1MHz-5MHz范围内的同步性能
    
    % 获取高频测试参数
    params = test_config.get_test_params('high');
    
    % 获取结果保存路径
    current_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    results_dir = fullfile(current_dir, 'results');
    plots_dir = fullfile(results_dir, 'plots', 'high');
    
    % 确保结果目录存在
    if ~exist(plots_dir, 'dir')
        mkdir(plots_dir);
    end
    
    % 初始化结果结构
    results = struct();
    results.freq_range = params.freq_range;
    results.snr_range = params.snr_range;
    results.algorithms = {'square_law', 'costas_loop', 'improved_costas', 'multi_stage'};
    
    % 为每个算法分配结果存储空间
    for alg = results.algorithms
        results.(alg{1}) = struct(...
            'freq_error', zeros(length(params.snr_range), params.num_trials), ...
            'conv_time', zeros(length(params.snr_range), params.num_trials), ...
            'cpu_time', zeros(length(params.snr_range), params.num_trials) ...
        );
    end
    
    % 测试每个SNR点
    for snr_idx = 1:length(params.snr_range)
        snr = params.snr_range(snr_idx);
        fprintf('测试SNR = %d dB\n', snr);
        
        % 对每个测试进行多次重复
        for trial = 1:params.num_trials
            % 随机选择载波频率
            f_carrier = params.freq_range.min_freq + ...
                       (params.freq_range.max_freq - params.freq_range.min_freq) * rand();
            
            % 生成测试信号
            t = 0:1/params.freq_range.sample_rate:params.test_duration;
            signal = generate_test_signal(t, f_carrier, snr, params.modulation_type);
            
            % 测试每个算法
            for alg = results.algorithms
                try
                    % 测量CPU时间
                    tic;
                    switch alg{1}
                        case 'square_law'
                            [freq_error, ~, debug_info] = square_law_sync(signal, params.freq_range.sample_rate, f_carrier);
                        case 'costas_loop'
                            [freq_error, ~, debug_info] = costas_loop_sync(signal, params.freq_range.sample_rate, f_carrier);
                        case 'improved_costas'
                            [freq_error, ~, debug_info] = improved_costas_sync(signal, params.freq_range.sample_rate, f_carrier);
                        case 'multi_stage'
                            [freq_error, ~, debug_info] = multi_stage_costas_sync(signal, params.freq_range.sample_rate, f_carrier);
                    end
                    cpu_time = toc;
                    
                    % 存储结果
                    results.(alg{1}).freq_error(snr_idx, trial) = abs(freq_error);
                    results.(alg{1}).cpu_time(snr_idx, trial) = cpu_time;
                    
                    % 为每个算法生成同步过程图
                    if trial == 1  % 只为第一次试验绘图
                        fig = figure('Visible', 'off');
                        plot_sync_process(debug_info, f_carrier, snr, t);
                        saveas(fig, fullfile(plots_dir, sprintf('%s_snr%d_trial%d.png', ...
                            alg{1}, snr, trial)));
                        close(fig);
                    end
                    
                catch e
                    warning('算法 %s 在高频测试中失败: %s', alg{1}, e.message);
                    % 记录错误结果
                    results.(alg{1}).freq_error(snr_idx, trial) = NaN;
                    results.(alg{1}).cpu_time(snr_idx, trial) = NaN;
                end
            end
        end
    end
    
    % 保存结果
    save(fullfile(results_dir, 'high_freq_results.mat'), 'results');
    generate_report(results, fullfile(results_dir, 'high_freq_report.txt'));
end

function signal = generate_test_signal(t, f_carrier, snr, modulation_type)
    % 生成测试信号
    switch upper(modulation_type)
        case 'BPSK'
            % 生成随机比特流
            bits = randi([0 1], 1, ceil(length(t)/100));
            % 上采样
            symbol_rate = length(t)/length(bits);
            symbols = repelem(2*bits-1, ceil(symbol_rate));
            symbols = symbols(1:length(t));
            % 调制
            signal = symbols .* cos(2*pi*f_carrier*t);
            % 添加噪声
            signal = awgn(signal, snr, 'measured');
        otherwise
            error('不支持的调制方式：%s', modulation_type);
    end
end

function save_results(results, filename)
    % 保存测试结果
    save(filename, 'results');
end

function generate_report(results, filename)
    % 生成测试报告
    fid = fopen(filename, 'w');
    fprintf(fid, '高频载波同步测试报告\n');
    fprintf(fid, '测试频率范围：%.2f MHz - %.2f MHz\n', ...
        results.freq_range.min_freq/1e6, results.freq_range.max_freq/1e6);
    fprintf(fid, '采样率：%.2f MHz\n\n', results.freq_range.sample_rate/1e6);
    
    % 输出每个算法的性能统计
    for alg = results.algorithms
        fprintf(fid, '\n%s算法性能统计：\n', alg{1});
        for snr_idx = 1:length(results.snr_range)
            snr = results.snr_range(snr_idx);
            freq_errors = results.(alg{1}).freq_error(snr_idx, :);
            cpu_times = results.(alg{1}).cpu_time(snr_idx, :);
            
            fprintf(fid, 'SNR = %d dB:\n', snr);
            fprintf(fid, '  频率误差均值：%.2f Hz\n', mean(freq_errors, 'omitnan'));
            fprintf(fid, '  频率误差标准差：%.2f Hz\n', std(freq_errors, 'omitnan'));
            fprintf(fid, '  平均CPU时间：%.2f ms\n', mean(cpu_times, 'omitnan')*1000);
        end
    end
    
    fclose(fid);
end

function plot_sync_process(debug_info, f_carrier, snr, t)
    % 绘制同步过程
    subplot(4,1,1);
    plot(t, debug_info.freq_history);
    hold on;
    plot([t(1), t(end)], [f_carrier, f_carrier], 'r--');
    title(sprintf('频率估计过程 (f=%.1fMHz, SNR=%.0fdB)', f_carrier/1e6, snr));
    xlabel('时间 (s)');
    ylabel('频率 (Hz)');
    legend('估计值', '实际值');
    grid on;
    
    subplot(4,1,2);
    plot(t, debug_info.phase_history);
    title('相位估计过程');
    xlabel('时间 (s)');
    ylabel('相位 (rad)');
    grid on;
    
    subplot(4,1,3);
    plot(t, debug_info.noise_bw_history);
    title('环路带宽变化');
    xlabel('时间 (s)');
    ylabel('带宽');
    grid on;
    
    subplot(4,1,4);
    plot(t, debug_info.error_signal);
    title('误差信号');
    xlabel('时间 (s)');
    ylabel('误差');
    grid on;
    
    % 添加预同步信息
    annotation('textbox', [0.15, 0.95, 0.7, 0.05], ...
        'String', sprintf('预同步结果: 频率误差=%.2fHz, SNR=%.1fdB', ...
        debug_info.initial_freq_error, debug_info.initial_snr), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
end 