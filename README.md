# 载波同步算法实现与测试

本项目实现了多种载波同步算法，并提供了完整的测试框架和性能对比分析。

## 项目结构

```
.
├── src/
│   ├── sync/                # 同步算法实现
│   │   ├── square_law_sync.m        # 平方律法
│   │   ├── costas_loop_sync.m       # 原始Costas环
│   │   ├── improved_costas_sync.m   # 改进的Costas环
│   │   └── multi_stage_costas_sync.m# 多级同步器
│   └── optimization/        # 优化算法
│       └── optimize_costas_params.m  # Costas环参数优化
├── test/
│   └── sync/               # 测试代码
│       └── test_sync_methods.m      # 同步算法测试脚本
├── utils/
│   └── common/            # 通用工具函数
├── main.m                 # 主程序入口
├── startup.m             # MATLAB环境设置脚本
└── README.md             # 项目说明文档
```

## 功能特性

1. **同步算法实现**
   - 平方律法：适用于BPSK信号的载波同步
   - 原始Costas环：经典的载波同步算法
   - 改进的Costas环：优化了环路参数和性能
   - 多级同步器：结合多种方法的高性能同步器

2. **参数优化**
   - 提供了Costas环参数的自动优化功能
   - 支持不同信噪比条件下的参数优化

3. **性能测试**
   - 完整的测试框架
   - 支持多种性能指标对比
   - 自动生成测试报告

## 使用方法

1. **环境设置**
   ```matlab
   >> startup  % 运行启动脚本，设置MATLAB环境
   ```

2. **运行测试**
   方法一：直接运行主程序
   ```matlab
   >> main
   ```
   
   方法二：单独运行测试脚本
   ```matlab
   >> test_sync_methods
   ```

3. **查看结果**
   - 测试结果将自动保存在 `sync_results.txt` 文件中
   - 使用 `concat_matlab_files.ps1` 可生成包含所有代码和结果的markdown文档

## 开发指南

1. **添加新的同步算法**
   - 在 `src/sync/` 目录下创建新的算法文件
   - 遵循现有的函数接口规范
   - 在测试脚本中添加新算法的测试用例

2. **修改测试参数**
   - 编辑 `test/sync/test_sync_methods.m` 文件
   - 可调整信噪比范围、测试次数等参数

3. **优化算法参数**
   - 使用 `src/optimization/optimize_costas_params.m` 进行参数优化
   - 可根据具体应用场景调整优化目标

## 注意事项

1. 首次使用时请先运行 `startup.m` 设置环境
2. 确保MATLAB版本兼容性（建议使用R2019b或更高版本）
3. 如遇到路径问题，可使用 `restoredefaultpath` 恢复默认路径后重新运行 `startup.m`

## 结果分析

测试结果将包含以下内容：
- 各算法在不同信噪比下的性能对比
- 收敛时间分析
- 频率估计精度评估
- 算法复杂度对比

详细的测试结果可在 `sync_results.txt` 中查看。 