# 环境说明

建议Python 3.12。requirements.txt给的是兼容安装范围，发布整理时不将其标为全部主实验的历史精确锁文件。

核心只需NumPy、SciPy、scikit-learn、pandas与threadpoolctl，无需PyTorch、CUDA或预训练权重。scikit-learn上限设为1.11以下，以保留原SVC接口；未因整理代码而调整模型参数。

已有直接完整版本清单仅对应CIC46标签量诊断：Python 3.12.10、NumPy 2.5.4、SciPy 1.18.1、scikit-learn 1.9.1、threadpoolctl 3.7.0。此证据另存environment/cic46_diagnostic_historical_environment.json，不能作为TON、RT、CIC39每轮的完整历史环境。

主实验完整逐轮依赖清单、CPU型号及物理内存未全部留存。已有记录显示核心使用单CPU线程，深度对照使用RTX 3050 Laptop 4 GB；本包没有重新执行这些深度对照。

doctor命令检查当前依赖实际能否导入并打印版本；run会把当前环境写到运行目录environment_at_run.json。该文件是新复现环境，不能覆盖或替代历史环境记录。发布时的检查记录见provenance/software_checks.json，其中只包含代码和入口检查，没有训练复验。
