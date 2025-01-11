% 主程序文件
% 运行载波同步系统测试

try
    % 获取当前脚本所在目录
    current_dir = fileparts(mfilename('fullpath'));
    
    % 添加所需路径
    addpath(genpath(fullfile(current_dir, 'src')));     % 添加所有源代码
    addpath(genpath(fullfile(current_dir, 'test')));    % 添加所有测试代码
    addpath(genpath(fullfile(current_dir, 'utils')));   % 添加所有工具函数
    addpath(genpath(fullfile(current_dir, 'config')));  % 添加配置文件
    
    % 清理工作空间
    clear;
    clc;
    
    % 重新获取目录路径（因为clear会清除所有变量）
    current_dir = fileparts(mfilename('fullpath'));
    results_dir = fullfile(current_dir, 'results');
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end
    
    % 运行低频测试
    fprintf('运行低频测试 (1kHz - 100kHz)...\n');
    low_freq_results = test_sync_methods('low');
    fprintf('低频测试完成\n');
    
    % 运行高频测试
    fprintf('\n运行高频测试 (1MHz - 5MHz)...\n');
    high_freq_results = test_sync_methods('high');
    fprintf('高频测试完成\n');
    
    % 生成综合报告
    fprintf('\n生成综合报告...\n');
    generate_combined_report(low_freq_results, high_freq_results);
    fprintf('测试完成！结果已保存到 %s 目录\n', results_dir);
    
catch e
    fprintf('错误：%s\n', e.message);
    fprintf('在文件 %s 第 %d 行\n', e.stack(1).file, e.stack(1).line);
end 