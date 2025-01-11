classdef sync_utils
    % 同步算法共享工具函数
    
    methods(Static)
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
        
        function plot_sync_process(debug_info, f_carrier, snr, t)
            % 创建新图形窗口并设置样式（不显示）
            fig = figure('Color', 'white', 'Position', [100, 100, 900, 800], ...
                'Visible', 'off', 'CreateFcn', '');
            
            % 设置默认字体
            set(fig, 'DefaultAxesFontSize', 10);
            set(fig, 'DefaultTextFontSize', 10);
            set(fig, 'DefaultAxesFontName', 'Microsoft YaHei');
            set(fig, 'DefaultTextFontName', 'Microsoft YaHei');
            
            % 频率估计过程
            subplot(4,1,1);
            plot(t, debug_info.freq_history, 'LineWidth', 1.5, 'Color', [0.2 0.4 0.8]);
            hold on;
            plot([t(1), t(end)], [f_carrier, f_carrier], 'r--', 'LineWidth', 1.2);
            if f_carrier >= 1e6
                title(sprintf('频率估计过程 (f=%.2f MHz, SNR=%.1f dB)', f_carrier/1e6, snr), ...
                    'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            else
                title(sprintf('频率估计过程 (f=%.2f kHz, SNR=%.1f dB)', f_carrier/1e3, snr), ...
                    'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            end
            xlabel('时间 (s)', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            ylabel('频率 (Hz)', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            legend({'估计值', '实际值'}, 'Location', 'best', 'Box', 'off', 'FontName', 'Microsoft YaHei');
            grid on;
            box on;
            
            % 相位估计过程
            subplot(4,1,2);
            plot(t, debug_info.phase_history, 'LineWidth', 1.5, 'Color', [0.8 0.4 0.2]);
            title('相位估计过程', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            xlabel('时间 (s)', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            ylabel('相位 (rad)', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            grid on;
            box on;
            
            % 带宽变化
            subplot(4,1,3);
            plot(t, debug_info.noise_bw_history, 'LineWidth', 1.5, 'Color', [0.2 0.6 0.4]);
            title('环路带宽变化', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            xlabel('时间 (s)', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            ylabel('带宽', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            grid on;
            box on;
            
            % 误差信号
            subplot(4,1,4);
            plot(t, debug_info.error_signal, 'LineWidth', 1.5, 'Color', [0.6 0.2 0.6]);
            title('误差信号', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            xlabel('时间 (s)', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            ylabel('误差', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            grid on;
            box on;
            
            % 添加预同步信息
            annotation('textbox', [0.15, 0.95, 0.7, 0.05], ...
                'String', sprintf('预同步结果: 频率误差=%.2f Hz, SNR=%.1f dB', ...
                debug_info.initial_freq_error, debug_info.initial_snr), ...
                'EdgeColor', 'none', ...
                'HorizontalAlignment', 'center', ...
                'FontWeight', 'bold', ...
                'FontName', 'Microsoft YaHei', ...
                'BackgroundColor', [0.95 0.95 0.95]);
            
            % 调整子图间距
            set(gcf, 'Units', 'normalized');
            set(findall(gcf, 'Type', 'axes'), 'Units', 'normalized');
            
            % 优化布局
            pos = get(gcf, 'Position');
            set(gcf, 'Position', [pos(1), pos(2), pos(3), pos(4)*1.1]);
            
            % 应用更改并保存
            drawnow;
            
            % 关闭图形
            close(fig);
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
        
        function [snr_estimate] = estimate_snr(signal, fs, f_carrier, freq_error)
            % 改进的SNR估计
            N = length(signal);
            steady_state_start = floor(N * 0.7);
            
            segment_length = floor(fs/10);  % 0.1秒segments
            num_segments = floor((N-steady_state_start+1)/segment_length);
            snr_estimates = zeros(1, num_segments);
            
            for i = 1:num_segments
                start_idx = steady_state_start + (i-1)*segment_length;
                end_idx = min(start_idx + segment_length - 1, N);
                segment = signal(start_idx:end_idx);
                
                % 使用Welch方法估计功率谱
                [pxx, f] = pwelch(segment, [], [], [], fs);
                
                % 自适应信号带宽
                center_freq = f_carrier + freq_error;
                signal_band = abs(f - center_freq) <= max(2, abs(freq_error)/5);
                
                signal_power = mean(pxx(signal_band));
                noise_power = mean(pxx(~signal_band));
                snr_estimates(i) = 10*log10(signal_power/noise_power);
            end
            
            % 去除异常值后平均
            snr_estimate = median(snr_estimates);
            snr_estimate = min(max(snr_estimate, 0), 40);  % 限制范围
        end
        
        function [K1, K2] = calculate_loop_coefficients(noise_bw, damping)
            % 计算环路滤波器系数
            K1 = 4 * damping * noise_bw / (1 + 2*damping*noise_bw + noise_bw^2);
            K2 = 4 * noise_bw^2 / (1 + 2*damping*noise_bw + noise_bw^2);
        end
        
        function freq = limit_frequency(freq, freq_max, fs)
            % 频率限幅
            freq_max_rad = 2*pi*freq_max/fs;
            freq = max(min(freq, freq_max_rad), -freq_max_rad);
        end
        
        function plot_performance_comparison(results, plots_dir, freq_mode)
            % 创建性能对比图（不显示）
            fig = figure('Name', '性能对比', 'Position', [100, 100, 1200, 800], ...
                'Color', 'white', 'Visible', 'off', 'CreateFcn', '');
            
            % 设置默认字体
            set(fig, 'DefaultAxesFontSize', 10);
            set(fig, 'DefaultTextFontSize', 10);
            set(fig, 'DefaultAxesFontName', 'Microsoft YaHei');
            set(fig, 'DefaultTextFontName', 'Microsoft YaHei');
            
            % 定义颜色方案
            colors = [0.2 0.4 0.8;   % 蓝色
                     0.8 0.4 0.2;   % 橙色
                     0.2 0.6 0.4;   % 绿色
                     0.6 0.2 0.6];  % 紫色
                     
            % 定义算法中文名称
            alg_names = containers.Map(...
                {'square_law', 'costas_loop', 'improved_costas', 'multi_stage'}, ...
                {'平方律法', '基础Costas环', '改进Costas环', '多级同步器'});
            
            % SNR vs 频率误差
            subplot(2,2,1);
            for i = 1:length(results.algorithms)
                alg = results.algorithms{i};
                mean_errors = mean(results.(alg).freq_error, 2, 'omitnan');
                std_errors = std(results.(alg).freq_error, 0, 2, 'omitnan');
                errorbar(results.snr_range, mean_errors, std_errors, 'o-', ...
                    'Color', colors(i,:), 'LineWidth', 1.5, ...
                    'MarkerFaceColor', colors(i,:), ...
                    'DisplayName', alg_names(alg));
                hold on;
            end
            xlabel('信噪比 SNR (dB)', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            ylabel('频率误差 (Hz)', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            title('频率估计性能对比', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            grid on;
            box on;
            legend('show', 'Location', 'best', 'Box', 'off', 'FontName', 'Microsoft YaHei');
            
            % SNR vs CPU时间
            subplot(2,2,2);
            for i = 1:length(results.algorithms)
                alg = results.algorithms{i};
                mean_times = mean(results.(alg).cpu_time, 2, 'omitnan') * 1000;
                plot(results.snr_range, mean_times, 'o-', ...
                    'Color', colors(i,:), 'LineWidth', 1.5, ...
                    'MarkerFaceColor', colors(i,:), ...
                    'DisplayName', alg_names(alg));
                hold on;
            end
            xlabel('信噪比 SNR (dB)', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            ylabel('处理时间 (ms)', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            title('算法计算复杂度分析', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            grid on;
            box on;
            legend('show', 'Location', 'best', 'Box', 'off', 'FontName', 'Microsoft YaHei');
            
            % 箱线图比较
            subplot(2,2,3);
            boxdata = [];
            labels = {};
            for i = 1:length(results.algorithms)
                alg = results.algorithms{i};
                boxdata = [boxdata, results.(alg).freq_error(:)];
                labels = [labels, repmat({alg_names(alg)}, 1, size(results.(alg).freq_error(:), 1))];
            end
            h = boxplot(boxdata, labels, 'Colors', colors, 'Symbol', '+r');
            set(h, 'LineWidth', 1.2);
            ylabel('频率误差 (Hz)', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            title('算法稳定性分析', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            grid on;
            box on;
            set(gca, 'FontName', 'Microsoft YaHei');
            % 添加箱线图说明
            annotation('textbox', [0.3, 0.25, 0.15, 0.1], ...
                'String', {'箱线图说明:', ...
                          '上边界: 75%分位数', ...
                          '中间线: 中位数', ...
                          '下边界: 25%分位数', ...
                          '+: 异常值'}, ...
                'EdgeColor', 'none', ...
                'FontName', 'Microsoft YaHei', ...
                'FontSize', 8);
            
            % 收敛时间分析
            subplot(2,2,4);
            for i = 1:length(results.algorithms)
                alg = results.algorithms{i};
                mean_conv = mean(results.(alg).conv_time, 2, 'omitnan') / ...
                    results.freq_range.sample_rate * 1000;
                plot(results.snr_range, mean_conv, 'o-', ...
                    'Color', colors(i,:), 'LineWidth', 1.5, ...
                    'MarkerFaceColor', colors(i,:), ...
                    'DisplayName', alg_names(alg));
                hold on;
            end
            xlabel('信噪比 SNR (dB)', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            ylabel('收敛时间 (ms)', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            title('算法收敛性能分析', 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            grid on;
            box on;
            legend('show', 'Location', 'best', 'Box', 'off', 'FontName', 'Microsoft YaHei');
            
            % 添加频率模式信息
            if strcmpi(freq_mode, 'high')
                sgtitle({sprintf('高频载波同步性能分析 (%.2f-%.2f MHz)', ...
                    results.freq_range.min_freq/1e6, ...
                    results.freq_range.max_freq/1e6), ...
                    sprintf('采样率: %.2f MHz', results.freq_range.sample_rate/1e6)}, ...
                    'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            else
                sgtitle({sprintf('低频载波同步性能分析 (%.2f-%.2f kHz)', ...
                    results.freq_range.min_freq/1e3, ...
                    results.freq_range.max_freq/1e3), ...
                    sprintf('采样率: %.2f kHz', results.freq_range.sample_rate/1e3)}, ...
                    'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');
            end
            
            % 添加测试信息
            annotation('textbox', [0.7, 0.02, 0.25, 0.05], ...
                'String', sprintf('测试时间: %s', datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), ...
                'EdgeColor', 'none', ...
                'HorizontalAlignment', 'right', ...
                'FontName', 'Microsoft YaHei', ...
                'FontSize', 8);
            
            % 优化布局
            set(gcf, 'Units', 'normalized');
            set(findall(gcf, 'Type', 'axes'), 'Units', 'normalized');
            
            % 获取plots根目录
            plots_root = fileparts(plots_dir);
            
            % 保存图形到plots根目录
            if strcmpi(freq_mode, 'high')
                saveas(fig, fullfile(plots_root, 'high_freq_performance.png'));
                saveas(fig, fullfile(plots_root, 'high_freq_performance.fig'));
                fprintf('性能对比图已保存到: %s\n', fullfile(plots_root, 'high_freq_performance.png'));
            else
                saveas(fig, fullfile(plots_root, 'low_freq_performance.png'));
                saveas(fig, fullfile(plots_root, 'low_freq_performance.fig'));
                fprintf('性能对比图已保存到: %s\n', fullfile(plots_root, 'low_freq_performance.png'));
            end
            
            % 关闭图形
            close(fig);
        end
    end
end 