% MATLAB启动脚本
% 设置项目环境

try
    % 获取当前脚本所在目录
    current_dir = fileparts(mfilename('fullpath'));
    
    % 定义所需的目录
    required_dirs = {'src', 'test', 'utils', 'config'};
    
    % 验证目录结构
    for i = 1:length(required_dirs)
        dir_path = fullfile(current_dir, required_dirs{i});
        if ~exist(dir_path, 'dir')
            error('缺少必要的目录：%s', required_dirs{i});
        end
    end
    
    % 添加所有必要的路径
    addpath(genpath(fullfile(current_dir, 'src')));     % 源代码
    addpath(genpath(fullfile(current_dir, 'test')));    % 测试代码
    addpath(genpath(fullfile(current_dir, 'utils')));   % 工具函数
    addpath(genpath(fullfile(current_dir, 'config')));  % 配置文件
    
    % 确保结果目录存在
    results_dir = fullfile(current_dir, 'results');
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
        fprintf('创建结果目录：%s\n', results_dir);
    end
    
    % 确保plots目录存在
    plots_dir = fullfile(results_dir, 'plots');
    if ~exist(plots_dir, 'dir')
        mkdir(plots_dir);
        mkdir(fullfile(plots_dir, 'low'));
        mkdir(fullfile(plots_dir, 'high'));
        fprintf('创建绘图目录：%s\n', plots_dir);
    end
    
    % 显示路径设置信息
    fprintf('\n项目路径已设置：\n');
    fprintf('- 源代码: %s\n', fullfile(current_dir, 'src'));
    fprintf('- 测试代码: %s\n', fullfile(current_dir, 'test'));
    fprintf('- 工具函数: %s\n', fullfile(current_dir, 'utils'));
    fprintf('- 配置文件: %s\n', fullfile(current_dir, 'config'));
    fprintf('- 结果目录: %s\n', results_dir);
    
    % 显示版本信息
    fprintf('\nMATLAB版本: %s\n', version);
    fprintf('项目就绪！\n');
    fprintf('\n使用说明：\n');
    fprintf('1. 运行 main 开始完整测试\n');
    fprintf('2. 使用 test_sync_methods(''low'') 运行低频测试\n');
    fprintf('3. 使用 test_sync_methods(''high'') 运行高频测试\n');
    
catch e
    fprintf('错误：%s\n', e.message);
    fprintf('在文件 %s 第 %d 行\n', e.stack(1).file, e.stack(1).line);
end 