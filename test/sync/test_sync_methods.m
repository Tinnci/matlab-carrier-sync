%{
载波同步方法测试脚本 - 第二版

测试目标：
1. 对比四种方法的性能：
   - 平方律法
   - 原始Costas环
   - 改进的Costas环
   - 多级同步器

2. 测试条件：
   - 频率偏移：0.5~50 Hz（扩大范围）
   - SNR：10~30 dB
   - 信号长度：10秒

3. 评估指标：
   - 频率估计精度
   - SNR估计准确度
   - 捕获时间
   - 计算复杂度

4. 可视化：
   - 频率估计误差随频偏变化
   - SNR估计误差分布
   - 同步过程动态特性
%}

function results = test_sync_methods(freq_mode)
    % 载波同步方法测试脚本
    % 支持低频和高频测试模式
    %
    % 输入参数：
    %   freq_mode - 'low'（默认）或 'high'，指定测试频率范围
    
    if nargin < 1
        freq_mode = 'low';
    end
    
    % 获取测试参数
    params = test_config.get_test_params(freq_mode);
    
    % 获取结果保存路径
    current_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    results_dir = fullfile(current_dir, 'results');
    plots_dir = fullfile(results_dir, 'plots', freq_mode);
    
    % 创建保存目录
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
            signal = sync_utils.generate_test_signal(t, f_carrier, snr, params.modulation_type);
            
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
                    results.(alg{1}).conv_time(snr_idx, trial) = debug_info.conv_time;
                    
                    % 为每个算法生成同步过程图
                    if trial == 1  % 只为第一次试验绘图
                        fig = figure('Visible', 'off');
                        sync_utils.plot_sync_process(debug_info, f_carrier, snr, t);
                        saveas(fig, fullfile(plots_dir, sprintf('%s_snr%d_trial%d.png', ...
                            alg{1}, snr, trial)));
                        close(fig);
                    end
                catch e
                    warning('算法 %s 在测试中失败: %s', alg{1}, e.message);
                    % 记录错误结果
                    results.(alg{1}).freq_error(snr_idx, trial) = NaN;
                    results.(alg{1}).cpu_time(snr_idx, trial) = NaN;
                    results.(alg{1}).conv_time(snr_idx, trial) = NaN;
                end
            end
        end
    end
    
    % 保存结果
    save(fullfile(results_dir, sprintf('%s_freq_results.mat', freq_mode)), 'results');
    generate_report(results, fullfile(results_dir, sprintf('%s_freq_report.txt', freq_mode)), freq_mode);
    
    % 生成性能对比图
    sync_utils.plot_performance_comparison(results, plots_dir, freq_mode);
end

function generate_report(results, filename, freq_mode)
    % 生成测试报告
    fid = fopen(filename, 'w');
    if strcmpi(freq_mode, 'high')
        fprintf(fid, '高频载波同步测试报告\n');
        fprintf(fid, '测试频率范围：%.2f MHz - %.2f MHz\n', ...
            results.freq_range.min_freq/1e6, results.freq_range.max_freq/1e6);
        fprintf(fid, '采样率：%.2f MHz\n\n', results.freq_range.sample_rate/1e6);
    else
        fprintf(fid, '低频载波同步测试报告\n');
        fprintf(fid, '测试频率范围：%.2f kHz - %.2f kHz\n', ...
            results.freq_range.min_freq/1e3, results.freq_range.max_freq/1e3);
        fprintf(fid, '采样率：%.2f kHz\n\n', results.freq_range.sample_rate/1e3);
    end
    
    % 输出每个算法的性能统计
    for alg = results.algorithms
        fprintf(fid, '\n%s算法性能统计：\n', alg{1});
        for snr_idx = 1:length(results.snr_range)
            snr = results.snr_range(snr_idx);
            freq_errors = results.(alg{1}).freq_error(snr_idx, :);
            cpu_times = results.(alg{1}).cpu_time(snr_idx, :);
            conv_times = results.(alg{1}).conv_time(snr_idx, :);
            
            fprintf(fid, 'SNR = %d dB:\n', snr);
            fprintf(fid, '  频率误差均值：%.2f Hz\n', mean(freq_errors, 'omitnan'));
            fprintf(fid, '  频率误差标准差：%.2f Hz\n', std(freq_errors, 'omitnan'));
            fprintf(fid, '  平均CPU时间：%.2f ms\n', mean(cpu_times, 'omitnan')*1000);
            fprintf(fid, '  平均收敛时间：%.2f ms\n', mean(conv_times, 'omitnan')/results.freq_range.sample_rate*1000);
        end
    end
    
    fclose(fid);
end

function plot_performance_comparison(results, plots_dir, freq_mode)
    % 创建性能对比图
    fig = figure('Name', '性能对比', 'Position', [100, 100, 1200, 800]);
    
    % SNR vs 频率误差
    subplot(2,2,1);
    for alg = results.algorithms
        mean_errors = mean(results.(alg{1}).freq_error, 2, 'omitnan');
        std_errors = std(results.(alg{1}).freq_error, 0, 2, 'omitnan');
        errorbar(results.snr_range, mean_errors, std_errors, 'o-', ...
            'DisplayName', alg{1});
        hold on;
    end
    xlabel('SNR (dB)');
    ylabel('频率误差 (Hz)');
    title('频率估计性能');
    grid on;
    legend('show');
    
    % SNR vs CPU时间
    subplot(2,2,2);
    for alg = results.algorithms
        mean_times = mean(results.(alg{1}).cpu_time, 2, 'omitnan') * 1000;
        plot(results.snr_range, mean_times, 'o-', 'DisplayName', alg{1});
        hold on;
    end
    xlabel('SNR (dB)');
    ylabel('处理时间 (ms)');
    title('计算复杂度');
    grid on;
    legend('show');
    
    % 箱线图比较
    subplot(2,2,3);
    boxdata = [];
    labels = {};
    for alg = results.algorithms
        boxdata = [boxdata, results.(alg{1}).freq_error(:)];
        labels = [labels, repmat({alg{1}}, 1, size(results.(alg{1}).freq_error(:), 1))];
    end
    boxplot(boxdata, labels);
    ylabel('频率误差 (Hz)');
    title('算法性能分布');
    grid on;
    
    % 收敛时间分析
    subplot(2,2,4);
    for alg = results.algorithms
        mean_conv = mean(results.(alg{1}).conv_time, 2, 'omitnan') / results.freq_range.sample_rate * 1000;
        plot(results.snr_range, mean_conv, 'o-', 'DisplayName', alg{1});
        hold on;
    end
    xlabel('SNR (dB)');
    ylabel('收敛时间 (ms)');
    title('收敛性能');
    grid on;
    legend('show');
    
    % 保存图形
    saveas(fig, fullfile(plots_dir, 'performance_comparison.png'));
    close(fig);
end 