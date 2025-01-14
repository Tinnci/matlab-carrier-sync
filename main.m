% main.m
% 主程序文件
% 运行载波同步系统测试

% 获取当前脚本所在目录
current_dir = fileparts(mfilename('fullpath'));

% 添加所需路径
addpath(fullfile(current_dir, 'src', 'sync'));
addpath(fullfile(current_dir, 'src', 'optimization'));
addpath(fullfile(current_dir, 'test', 'sync'));

% 清理工作空间
clear;
clc;

% 运行测试
test_sync_methods(); 
