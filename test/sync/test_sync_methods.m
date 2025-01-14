% test_sync_methods.m
function test_sync_methods()
    % 测试参数
    fs = 1000;          % 采样频率
    f_carrier = 100;    % 载波频率
    signal_length = 10; % 信号长度（秒）

    % 扩展测试条件
    freq_offsets = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150];  % 扩展到150Hz
    snrs = [10, 20, 30];  % dB

    % 调制方式
    modulation_types = {'AM', 'BPSK', 'QPSK', '16-QAM'};

    % 创建结果文件
    fid = fopen('sync_results.txt', 'w');
    fprintf(fid, '载波同步系统测试结果（第三版）\n');
    fprintf(fid, '===================\n\n');

    % 运行参数优化并获取最优参数
    fprintf('开始参数优化...\n');
    best_params = optimize_costas_params();  % 确保您有此函数
    fprintf('参数优化完成。\n\n');

    % 创建性能统计结构
    stats = initialize_statistics();

    % 获取中文显示名称
    display_names = stats.method_display_names;

    % 创建图形窗口
    figure('Name', '同步性能对比', 'Position', [100, 100, 1600, 900]);

    % 对每种测试条件进行测试
    for mod_type = modulation_types
        current_mod = mod_type{1};
        fprintf(fid, '=== 调制方式: %s ===\n\n', current_mod);

        for f_offset = freq_offsets
            for snr = snrs
                % 生成测试信号
                N = floor(fs * signal_length);  % 确保采样点数量正确
                t = (0:N-1)/fs;  % 生成对应的时间向量
                switch current_mod
                    case 'AM'
                        modulated_signal = cos(2*pi*(f_carrier + f_offset)*t);
                    case 'BPSK'
                        data = randi([0 1], 1, N);
                        bpsk_signal = 2*data - 1;
                        modulated_signal = bpsk_signal .* cos(2*pi*(f_carrier + f_offset)*t);
                    case 'QPSK'
                        data = randi([0 3], 1, N);
                        qpsk_signal = exp(1j * pi/2 * data);
                        modulated_signal = real(qpsk_signal .* exp(1j*2*pi*(f_carrier + f_offset)*t));
                    case '16-QAM'
                        data = randi([0 15], 1, N);
                        qam_signal = qammod(data, 16, 'UnitAveragePower', true);
                        modulated_signal = real(qam_signal .* exp(1j*2*pi*(f_carrier + f_offset)*t));
                    otherwise
                        error('未知的调制方式: %s', current_mod);
                end
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

                % 5. PLL同步法
                tic;
                [freq_error5, snr_est5, debug5] = pll_sync(noisy_signal, fs, f_carrier, 1, 0.707, best_params.freq_max);
                time5 = toc;

                % 6. 粒子滤波器同步法
                tic;
                [freq_error6, snr_est6, debug6] = particle_filter_sync(noisy_signal, fs, f_carrier, 100, best_params.freq_max);
                time6 = toc;

                % 记录结果
                fprintf(fid, '测试条件:\n');
                fprintf(fid, '调制方式: %s\n', current_mod);
                fprintf(fid, '频率偏差: %.1f Hz\n', f_offset);
                fprintf(fid, 'SNR: %.0f dB\n\n', snr);

                % 记录各方法结果
                methods = {'square_law', 'original_costas', 'improved_costas', 'multi_stage', 'pll', 'particle_filter'};
                freq_errors = [freq_error1, freq_error2, freq_error3, freq_error4, freq_error5, freq_error6];
                snr_ests = [snr_est1, snr_est2, snr_est3, snr_est4, snr_est5, snr_est6];
                times = [time1, time2, time3, time4, time5, time6];

                for i = 1:length(methods)
                    if f_offset ~= 0
                        freq_err_percent = abs((freq_errors(i)-f_offset)/f_offset) * 100;
                    else
                        freq_err_percent = 0;
                    end
                    fprintf(fid, '%s结果:\n', display_names{i});
                    fprintf(fid, '估计频率误差: %.2f Hz\n', freq_errors(i));
                    fprintf(fid, '估计SNR: %.2f dB\n', snr_ests(i));
                    fprintf(fid, '频率误差精度: %.2f%%\n', freq_err_percent);
                    fprintf(fid, '处理时间: %.3f 秒\n\n', times(i));

                    % 更新统计信息
                    stats = update_statistics(stats, methods{i}, display_names{i}, f_offset, snr, ...
                        freq_err_percent, abs(snr_ests(i)-snr), times(i));
                end

                fprintf(fid, '-------------------\n\n');

                % 绘制同步过程（仅绘制多级同步器）
                plot_sync_process(debug4, f_offset, snr, t, current_mod);
            end
        end
    end

    % 初始化统计结构
    function stats = initialize_statistics()
        stats = struct();

        % 使用英文名称作为字段名称
        methods = {'square_law', 'original_costas', 'improved_costas', 'multi_stage', 'pll', 'particle_filter'};

        % 对应的中文显示名称
        display_names = {'平方变换法', '原始Costas环法', '改进Costas环法', '多级同步法', 'PLL同步法', '粒子滤波器同步法'};

        for i = 1:length(methods)
            stats.(methods{i}).freq_errors = [];
            stats.(methods{i}).snr_errors = [];
            stats.(methods{i}).times = [];
        end

        % 将显示名称存储在结构体中
        stats.method_display_names = display_names;
    end

    % 更新统计信息
    function stats = update_statistics(stats, method, display_name, f_offset, snr, ...
        freq_error_percent, snr_error, time)
        stats.(method).freq_errors(end+1, :) = [f_offset, snr, freq_error_percent];
        stats.(method).snr_errors(end+1, :) = [f_offset, snr, snr_error];
        stats.(method).times(end+1, :) = [f_offset, snr, time];
    end

    % 输出统计摘要
    function print_statistics_summary(fid, stats, modulation_types)
        fprintf(fid, '\n性能统计摘要\n');
        fprintf(fid, '===================\n\n');

        methods = {'square_law', 'original_costas', 'improved_costas', 'multi_stage', 'pll', 'particle_filter'};
        display_names = stats.method_display_names;

        for i = 1:length(methods)
            method = methods{i};
            display_name = display_names{i};
            fprintf(fid, '%s:\n', display_name);

            % 频率误差统计
            freq_errs = stats.(method).freq_errors(:,3);
            fprintf(fid, '频率误差统计:\n');
            fprintf(fid, '  平均值: %.2f%%\n', mean(freq_errs));
            fprintf(fid, '  中位数: %.2f%%\n', median(freq_errs));
            fprintf(fid, '  最大值: %.2f%%\n', max(freq_errs));
            fprintf(fid, '  最小值: %.2f%%\n', min(freq_errs));
            fprintf(fid, '  标准差: %.2f%%\n\n', std(freq_errs));

            % SNR误差统计
            snr_errs = stats.(method).snr_errors(:,3);
            fprintf(fid, 'SNR误差统计:\n');
            fprintf(fid, '  平均值: %.2f dB\n', mean(snr_errs));
            fprintf(fid, '  中位数: %.2f dB\n', median(snr_errs));
            fprintf(fid, '  最大值: %.2f dB\n', max(snr_errs));
            fprintf(fid, '  最小值: %.2f dB\n', min(snr_errs));
            fprintf(fid, '  标准差: %.2f dB\n\n', std(snr_errs));

            % 处理时间统计
            times = stats.(method).times(:,3);
            fprintf(fid, '处理时间统计:\n');
            fprintf(fid, '  平均值: %.3f 秒\n', mean(times));
            fprintf(fid, '  中位数: %.3f 秒\n', median(times));
            fprintf(fid, '  最大值: %.3f 秒\n', max(times));
            fprintf(fid, '  最小值: %.3f 秒\n\n', min(times));
        end
    end

    % 绘制同步过程
    function plot_sync_process(debug_info, f_offset, snr, t, mod_type)
        clf;

        % 频率估计过程
        subplot(3,2,1);
        plot(t, debug_info.tracking_stage.freq_history);
        hold on;
        plot([t(1), t(end)], [f_offset, f_offset], 'r--');
        title(sprintf('频率估计过程 (%s, offset=%.1fHz, SNR=%.0fdB)', mod_type, f_offset, snr));
        xlabel('时间 (s)');
        ylabel('频率 (Hz)');
        legend('估计值', '实际值');
        grid on;

        % 相位估计过程
        subplot(3,2,2);
        plot(t, debug_info.tracking_stage.phase_history);
        title('相位估计过程');
        xlabel('时间 (s)');
        ylabel('相位 (rad)');
        grid on;

        % 误差信号
        subplot(3,2,3);
        plot(t, debug_info.tracking_stage.error_signal);
        title('误差信号');
        xlabel('时间 (s)');
        ylabel('误差');
        grid on;

        % 粒子分布（仅适用于粒子滤波器同步法）
        subplot(3,2,4);
        if isfield(debug_info.tracking_stage, 'particles') && isfield(debug_info.tracking_stage, 'weights')
            scatter(debug_info.tracking_stage.particles, debug_info.tracking_stage.weights, 10, 'filled');
            title('粒子滤波器分布');
            xlabel('频率 (Hz)');
            ylabel('权重');
            grid on;
        else
            text(0.5, 0.5, '无粒子滤波器数据', 'HorizontalAlignment', 'center');
            axis off;
        end

        % 频率历史
        subplot(3,2,5);
        plot(t, debug_info.tracking_stage.freq_history);
        hold on;
        plot([t(1), t(end)], [f_offset, f_offset], 'r--');
        title('频率历史');
        xlabel('时间 (s)');
        ylabel('频率 (Hz)');
        grid on;

        % SNR历史
        if isfield(debug_info.tracking_stage, 'snr_estimate')
            subplot(3,2,6);
            plot(t, debug_info.tracking_stage.snr_estimate);
            title('SNR历史');
            xlabel('时间 (s)');
            ylabel('SNR (dB)');
            grid on;
        else
            subplot(3,2,6);
            text(0.5, 0.5, '无SNR历史数据', 'HorizontalAlignment', 'center');
            axis off;
        end

        drawnow;
        pause(0.1);
    end

    % 绘制性能对比图
    function plot_performance_comparison(stats, freq_offsets, modulation_types)
        % 创建性能对比图
        figure('Name', '性能对比', 'Position', [100, 100, 1600, 900]);

        methods = {'square_law', 'original_costas', 'improved_costas', 'multi_stage', 'pll', 'particle_filter'};
        display_names = stats.method_display_names;

        % 频率误差 vs 频率偏移
        subplot(3,2,1);
        hold on;
        for i = 1:length(methods)
            data = stats.(methods{i}).freq_errors;
            plot(data(:,1), data(:,3), 'o-', 'DisplayName', display_names{i});
        end
        xlabel('频率偏移 (Hz)');
        ylabel('频率误差 (%)');
        title('频率估计性能');
        grid on;
        legend('show', 'Location', 'best');

        % SNR误差 vs 频率偏移
        subplot(3,2,2);
        hold on;
        for i = 1:length(methods)
            data = stats.(methods{i}).snr_errors;
            plot(data(:,1), data(:,3), 'o-', 'DisplayName', display_names{i});
        end
        xlabel('频率偏移 (Hz)');
        ylabel('SNR误差 (dB)');
        title('SNR估计性能');
        grid on;
        legend('show', 'Location', 'best');

        % 处理时间 vs 频率偏移
        subplot(3,2,3);
        hold on;
        for i = 1:length(methods)
            data = stats.(methods{i}).times;
            plot(data(:,1), data(:,3), 'o-', 'DisplayName', display_names{i});
        end
        xlabel('频率偏移 (Hz)');
        ylabel('处理时间 (s)');
        title('计算复杂度');
        grid on;
        legend('show', 'Location', 'best');

        % 频率误差箱线图
        subplot(3,2,4);
        all_freq_errors = [];
        group_labels_freq = {};
        for i = 1:length(methods)
            current_data = stats.(methods{i}).freq_errors(:,3);
            all_freq_errors = [all_freq_errors; current_data(:)];
            group_labels_freq = [group_labels_freq; repmat(display_names(i), length(current_data), 1)];
        end
        boxplot(all_freq_errors, group_labels_freq, 'Labels', display_names);
        ylabel('频率误差 (%)');
        title('频率误差分布');
        grid on;

        % SNR误差箱线图
        subplot(3,2,5);
        all_snr_errors = [];
        group_labels_snr = {};
        for i = 1:length(methods)
            current_data = stats.(methods{i}).snr_errors(:,3);
            all_snr_errors = [all_snr_errors; current_data(:)];
            group_labels_snr = [group_labels_snr; repmat(display_names(i), length(current_data), 1)];
        end
        boxplot(all_snr_errors, group_labels_snr, 'Labels', display_names);
        ylabel('SNR误差 (dB)');
        title('SNR误差分布');
        grid on;

        % 处理时间箱线图
        subplot(3,2,6);
        all_time_errors = [];
        group_labels_time = {};
        for i = 1:length(methods)
            current_data = stats.(methods{i}).times(:,3);
            all_time_errors = [all_time_errors; current_data(:)];
            group_labels_time = [group_labels_time; repmat(display_names(i), length(current_data), 1)];
        end
        boxplot(all_time_errors, group_labels_time, 'Labels', display_names);
        ylabel('处理时间 (s)');
        title('处理时间分布');
        grid on;

        drawnow;
    end
end
