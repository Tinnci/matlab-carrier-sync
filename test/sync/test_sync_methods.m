% test_sync_methods.m
function test_sync_methods()
    % 测试参数
    fs = 1000;          % 采样频率
    f_carrier = 100;    % 载波频率
    signal_length = 10; % 信号长度（秒）

    % 扩展测试条件
    freq_offsets = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];  % 扩展到100Hz
    snrs = [10, 20, 30];  % dB

    % 创建结果文件
    fid = fopen('sync_results.txt', 'w');
    fprintf(fid, '载波同步系统测试结果（第二版）\n');
    fprintf(fid, '===================\n\n');

    % 运行参数优化并获取最优参数
    fprintf('开始参数优化...\n');
    best_params = optimize_costas_params();  % 确保您有此函数
    fprintf('参数优化完成。\n\n');

    % 创建性能统计结构
    stats = initialize_statistics();

    % 创建图形窗口
    figure('Name', '同步性能对比', 'Position', [100, 100, 1200, 800]);

    % 对每种测试条件进行测试
    for f_offset = freq_offsets
        for snr = snrs
            % 生成测试信号
            N = floor(fs * signal_length);  % 确保采样点数量正确
            t = (0:N-1)/fs;  % 生成对应的时间向量
            modulated_signal = cos(2*pi*(f_carrier + f_offset)*t);
            noisy_signal = awgn(modulated_signal, snr, 'measured');

            % 测试各种方法
            % 1. 平方律法
            tic;
            [freq_error1, snr_est1] = square_law_sync(noisy_signal, fs, f_carrier);
            time1 = toc;

            % 2. 原始Costas环法（使用优化后的参数）
            tic;
            [freq_error2, snr_est2] = costas_loop_sync(noisy_signal, fs, f_carrier, ...
                best_params.noise_bw, best_params.damping, best_params.freq_max);
            time2 = toc;

            % 3. 改进Costas环法（使用优化后的参数）
            tic;
            [freq_error3, snr_est3, debug3] = improved_costas_sync(noisy_signal, fs, f_carrier, ...
                best_params.noise_bw, best_params.damping, best_params.freq_max);
            time3 = toc;

            % 4. 多级同步法（使用优化后的参数）
            tic;
            [freq_error4, snr_est4, debug4] = multi_stage_costas_sync(noisy_signal, fs, f_carrier, ...
                best_params.noise_bw, best_params.damping, best_params.freq_max);
            time4 = toc;

            % 记录结果
            fprintf(fid, '测试条件:\n');
            fprintf(fid, '频率偏差: %.1f Hz\n', f_offset);
            fprintf(fid, 'SNR: %.0f dB\n\n', snr);

            % 记录各方法结果
            methods = {'平方变换法', '原始Costas环法', '改进Costas环法', '多级同步法'};
            freq_errors = [freq_error1, freq_error2, freq_error3, freq_error4];
            snr_ests = [snr_est1, snr_est2, snr_est3, snr_est4];
            times = [time1, time2, time3, time4];

            for i = 1:length(methods)
                fprintf(fid, '%s结果:\n', methods{i});
                fprintf(fid, '估计频率误差: %.2f Hz\n', freq_errors(i));
                fprintf(fid, '估计SNR: %.2f dB\n', snr_ests(i));
                if f_offset ~= 0
                    freq_err_percent = abs((freq_errors(i)-f_offset)/f_offset) * 100;
                else
                    freq_err_percent = 0;
                end
                fprintf(fid, '频率误差精度: %.2f%%\n', freq_err_percent);
                fprintf(fid, '处理时间: %.3f 秒\n\n', times(i));

                % 更新统计信息
                stats = update_statistics(stats, i, f_offset, snr, ...
                    freq_err_percent, abs(snr_ests(i)-snr), times(i));
            end

            fprintf(fid, '-------------------\n\n');

            % 绘制同步过程（仅绘制多级同步器）
            plot_sync_process(debug4, f_offset, snr, t);
        end
    end

    % 输出统计摘要
    print_statistics_summary(fid, stats, methods);

    fclose(fid);

    % 绘制性能对比图
    plot_performance_comparison(stats, freq_offsets, methods);
end

function stats = initialize_statistics()
    % 初始化统计结构
    stats = struct();
    stats.freq_errors = cell(4,1);
    stats.snr_errors = cell(4,1);
    stats.times = cell(4,1);
    for i = 1:4
        stats.freq_errors{i} = [];
        stats.snr_errors{i} = [];
        stats.times{i} = [];
    end
end

function stats = update_statistics(stats, method_idx, f_offset, snr, ...
    freq_error, snr_error, time)
    % 更新统计信息
    stats.freq_errors{method_idx}(end+1,:) = [f_offset, snr, freq_error];
    stats.snr_errors{method_idx}(end+1,:) = [f_offset, snr, snr_error];
    stats.times{method_idx}(end+1,:) = [f_offset, snr, time];
end

function print_statistics_summary(fid, stats, methods)
    % 输出统计摘要
    fprintf(fid, '\n性能统计摘要\n');
    fprintf(fid, '===================\n\n');

    for i = 1:length(methods)
        fprintf(fid, '%s:\n', methods{i});

        % 频率误差统计
        freq_errs = stats.freq_errors{i}(:,3);
        fprintf(fid, '频率误差统计:\n');
        fprintf(fid, '  平均值: %.2f%%\n', mean(freq_errs));
        fprintf(fid, '  中位数: %.2f%%\n', median(freq_errs));
        fprintf(fid, '  最大值: %.2f%%\n', max(freq_errs));
        fprintf(fid, '  最小值: %.2f%%\n', min(freq_errs));
        fprintf(fid, '  标准差: %.2f%%\n\n', std(freq_errs));

        % SNR误差统计
        snr_errs = stats.snr_errors{i}(:,3);
        fprintf(fid, 'SNR误差统计:\n');
        fprintf(fid, '  平均值: %.2f dB\n', mean(snr_errs));
        fprintf(fid, '  中位数: %.2f dB\n', median(snr_errs));
        fprintf(fid, '  最大值: %.2f dB\n', max(snr_errs));
        fprintf(fid, '  最小值: %.2f dB\n', min(snr_errs));
        fprintf(fid, '  标准差: %.2f dB\n\n', std(snr_errs));

        % 处理时间统计
        times = stats.times{i}(:,3);
        fprintf(fid, '处理时间统计:\n');
        fprintf(fid, '  平均值: %.3f 秒\n', mean(times));
        fprintf(fid, '  中位数: %.3f 秒\n', median(times));
        fprintf(fid, '  最大值: %.3f 秒\n', max(times));
        fprintf(fid, '  最小值: %.3f 秒\n\n', min(times));
    end
end

function plot_sync_process(debug_info, f_offset, snr, t)
    % 绘制同步过程
    clf;

    % 频率估计过程
    subplot(3,1,1);
    plot(t, debug_info.tracking_stage.freq_history);
    hold on;
    plot([t(1), t(end)], [f_offset, f_offset], 'r--');
    title(sprintf('频率估计过程 (offset=%.1fHz, SNR=%.0fdB)', f_offset, snr));
    xlabel('时间 (s)');
    ylabel('频率 (Hz)');
    legend('估计值', '实际值');
    grid on;

    % 相位估计过程
    subplot(3,1,2);
    plot(t, debug_info.tracking_stage.phase_history);
    title('相位估计过程');
    xlabel('时间 (s)');
    ylabel('相位 (rad)');
    grid on;

    % 误差信号
    subplot(3,1,3);
    plot(t, debug_info.tracking_stage.error_signal);
    title('误差信号');
    xlabel('时间 (s)');
    ylabel('误差');
    grid on;

    % 添加多级同步信息
    annotation('textbox', [0.15, 0.95, 0.7, 0.05], ...
        'String', sprintf('多级同步过程: 频率误差=%.2fHz, SNR=%.2fdB', ...
        debug_info.tracking_stage.freq_error, snr), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    drawnow;
    pause(0.1);
end

function plot_performance_comparison(stats, freq_offsets, methods)
    % 创建性能对比图
    figure('Name', '性能对比', 'Position', [100, 100, 1200, 800]);

    % 频率误差vs频率偏移
    subplot(2,2,1);
    hold on;
    for i = 1:length(methods)
        data = stats.freq_errors{i};
        for snr_val = unique(data(:,2))'
            mask = data(:,2) == snr_val;
            plot(data(mask,1), data(mask,3), 'o-', 'DisplayName', ...
                sprintf('%s (SNR=%ddB)', methods{i}, snr_val));
        end
    end
    xlabel('频率偏移 (Hz)');
    ylabel('频率误差 (%)');
    title('频率估计性能');
    grid on;
    legend('show', 'Location', 'best');

    % SNR误差vs频率偏移
    subplot(2,2,2);
    hold on;
    for i = 1:length(methods)
        data = stats.snr_errors{i};
        for snr_val = unique(data(:,2))'
            mask = data(:,2) == snr_val;
            plot(data(mask,1), data(mask,3), 'o-', 'DisplayName', ...
                sprintf('%s (SNR=%ddB)', methods{i}, snr_val));
        end
    end
    xlabel('频率偏移 (Hz)');
    ylabel('SNR误差 (dB)');
    title('SNR估计性能');
    grid on;
    legend('show', 'Location', 'best');

    % 处理时间vs频率偏移
    subplot(2,2,3);
    hold on;
    for i = 1:length(methods)
        data = stats.times{i};
        plot(data(:,1), data(:,3), 'o-', 'DisplayName', methods{i});
    end
    xlabel('频率偏移 (Hz)');
    ylabel('处理时间 (s)');
    title('计算复杂度');
    grid on;
    legend('show', 'Location', 'best');

    % 箱线图比较
    subplot(2,2,4);
    % 重新组织数据用于箱线图
    all_errors = cell(1, length(methods));
    for i = 1:length(methods)
        all_errors{i} = stats.freq_errors{i}(:,3);
    end
    boxplot(cell2mat(all_errors), 'Labels', methods);
    ylabel('频率误差 (%)');
    title('方法间性能对比');
    grid on;
end
