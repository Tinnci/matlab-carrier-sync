function generate_combined_report(low_freq_results, high_freq_results)
    % 生成综合测试报告，对比低频和高频测试结果
    
    % 获取结果目录
    current_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    results_dir = fullfile(current_dir, 'results');
    
    % 确保结果目录存在
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    
    % 创建报告文件
    report_file = fullfile(results_dir, 'combined_sync_results.txt');
    fid = fopen(report_file, 'w');
    
    % 写入报告头
    fprintf(fid, '载波同步算法综合性能报告\n');
    fprintf(fid, '=======================\n\n');
    
    % 测试条件概述
    fprintf(fid, '测试条件:\n');
    fprintf(fid, '低频范围: %.2f kHz - %.2f kHz (采样率: %.2f MHz)\n', ...
        low_freq_results.freq_range.min_freq/1e3, ...
        low_freq_results.freq_range.max_freq/1e3, ...
        low_freq_results.freq_range.sample_rate/1e6);
    fprintf(fid, '高频范围: %.2f MHz - %.2f MHz (采样率: %.2f MHz)\n\n', ...
        high_freq_results.freq_range.min_freq/1e6, ...
        high_freq_results.freq_range.max_freq/1e6, ...
        high_freq_results.freq_range.sample_rate/1e6);
    
    % 对每个算法进行分析
    algorithms = low_freq_results.algorithms;
    for alg = algorithms
        fprintf(fid, '\n%s算法性能分析:\n', alg{1});
        fprintf(fid, '-----------------\n');
        
        % 低频性能
        fprintf(fid, '低频性能:\n');
        analyze_performance(fid, low_freq_results.(alg{1}), ...
            low_freq_results.snr_range, 'Hz');
        
        % 高频性能
        fprintf(fid, '\n高频性能:\n');
        analyze_performance(fid, high_freq_results.(alg{1}), ...
            high_freq_results.snr_range, 'kHz');
        
        % 性能对比分析
        fprintf(fid, '\n性能对比分析:\n');
        compare_performance(fid, low_freq_results.(alg{1}), ...
            high_freq_results.(alg{1}));
        
        fprintf(fid, '\n');
    end
    
    % 算法间综合对比
    fprintf(fid, '\n算法综合对比\n');
    fprintf(fid, '============\n\n');
    
    % 计算并显示各算法在不同频率范围的综合得分
    calculate_overall_scores(fid, low_freq_results, high_freq_results);
    
    % 关闭文件
    fclose(fid);
    
    % 在命令窗口显示报告位置
    fprintf('综合报告已生成: %s\n', report_file);
end

function analyze_performance(fid, results, snr_range, unit)
    % 分析单个频率范围的性能
    
    % 计算平均性能指标
    mean_freq_error = mean(results.freq_error, 2, 'omitnan');
    std_freq_error = std(results.freq_error, 0, 2, 'omitnan');
    mean_cpu_time = mean(results.cpu_time, 2, 'omitnan') * 1000; % 转换为ms
    
    % 输出性能数据
    for i = 1:length(snr_range)
        fprintf(fid, '  SNR = %d dB:\n', snr_range(i));
        fprintf(fid, '    频率误差: %.2f ± %.2f %s\n', ...
            mean_freq_error(i), std_freq_error(i), unit);
        fprintf(fid, '    处理时间: %.2f ms\n', mean_cpu_time(i));
    end
end

function compare_performance(fid, low_freq, high_freq)
    % 对比低频和高频性能
    
    % 计算平均处理时间比
    low_time = mean(low_freq.cpu_time(:), 'omitnan');
    high_time = mean(high_freq.cpu_time(:), 'omitnan');
    time_ratio = high_time / low_time;
    
    % 计算平均频率误差比
    low_error = mean(low_freq.freq_error(:), 'omitnan');
    high_error = mean(high_freq.freq_error(:), 'omitnan');
    error_ratio = high_error / low_error;
    
    % 输出对比结果
    fprintf(fid, '  处理时间比(高/低): %.2f\n', time_ratio);
    fprintf(fid, '  频率误差比(高/低): %.2f\n', error_ratio);
    
    % 性能建议
    if time_ratio > 2
        fprintf(fid, '  建议: 在高频应用中考虑优化算法效率\n');
    end
    if error_ratio > 2
        fprintf(fid, '  建议: 在高频应用中需要改进频率估计精度\n');
    end
end

function calculate_overall_scores(fid, low_freq_results, high_freq_results)
    % 计算各算法的综合得分
    
    algorithms = low_freq_results.algorithms;
    scores = zeros(length(algorithms), 2); % [低频得分, 高频得分]
    
    for i = 1:length(algorithms)
        alg = algorithms{i};
        
        % 计算低频得分
        low_score = calculate_score(low_freq_results.(alg));
        
        % 计算高频得分
        high_score = calculate_score(high_freq_results.(alg));
        
        scores(i,:) = [low_score, high_score];
        
        % 输出得分
        fprintf(fid, '%s算法:\n', alg);
        fprintf(fid, '  低频得分: %.2f\n', low_score);
        fprintf(fid, '  高频得分: %.2f\n', high_score);
        fprintf(fid, '  综合得分: %.2f\n\n', mean([low_score, high_score]));
    end
    
    % 输出最佳算法建议
    [~, best_low] = max(scores(:,1));
    [~, best_high] = max(scores(:,2));
    
    fprintf(fid, '算法建议:\n');
    fprintf(fid, '低频应用推荐: %s\n', algorithms{best_low});
    fprintf(fid, '高频应用推荐: %s\n', algorithms{best_high});
end

function score = calculate_score(results)
    % 计算单个算法的性能得分
    % 得分基于频率误差、处理时间和稳定性
    
    % 计算平均频率误差（归一化）
    mean_error = mean(results.freq_error(:), 'omitnan');
    error_score = 100 * exp(-mean_error/100);
    
    % 计算平均处理时间（归一化）
    mean_time = mean(results.cpu_time(:), 'omitnan');
    time_score = 100 * exp(-mean_time);
    
    % 计算稳定性（基于标准差）
    stability = std(results.freq_error(:), 'omitnan');
    stability_score = 100 * exp(-stability/50);
    
    % 综合得分（权重可调）
    score = 0.4 * error_score + 0.3 * time_score + 0.3 * stability_score;
end 