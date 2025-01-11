classdef test_config
    properties (Constant)
        % 低频测试配置
        LOW_FREQ_RANGE = struct(...
            'min_freq', 1e3, ...    % 1kHz
            'max_freq', 100e3, ...  % 100kHz
            'sample_rate', 1e6);    % 1MHz 采样率
        
        % 高频测试配置
        HIGH_FREQ_RANGE = struct(...
            'min_freq', 1e6, ...    % 1MHz
            'max_freq', 5e6, ...    % 5MHz
            'sample_rate', 50e6);   % 50MHz 采样率
        
        % 通用测试参数
        COMMON_PARAMS = struct(...
            'snr_range', -10:2:10, ...  % SNR范围
            'test_duration', 1e-3, ...   % 测试持续时间
            'num_trials', 100, ...       % 每个条件的测试次数
            'modulation_type', 'BPSK');  % 调制方式
        
        % 算法特定参数
        ALGORITHM_PARAMS = struct(...
            'costas_loop', struct('damping', 0.707, 'noise_bw', 0.1), ...
            'improved_costas', struct('damping', 0.707, 'noise_bw', 0.05), ...
            'square_law', struct('filter_order', 4), ...
            'multi_stage', struct('stages', 3, 'damping', [0.707, 0.8, 0.9]));
    end
    
    methods (Static)
        function params = get_test_params(freq_mode)
            % 根据频率模式返回测试参数
            params = test_config.COMMON_PARAMS;
            
            switch lower(freq_mode)
                case 'low'
                    params.freq_range = test_config.LOW_FREQ_RANGE;
                case 'high'
                    params.freq_range = test_config.HIGH_FREQ_RANGE;
                otherwise
                    error('未知的频率模式：%s', freq_mode);
            end
        end
    end
end 