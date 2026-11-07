# 有限新增标注下的物联网稀有攻击类别补检：核心复现代码

版本0.1.1，发布整理日期2026年11月7日。作者：孙盛祖、于万钧。通讯邮箱：256092477@mail.sit.edu.cn。

本仓库整理已有的普通树模型tree70与完整MAT-PV流程，供取得相同数据版本后复现。完整流程包含极端随机树、图残差纠错、高斯统计记忆、二分类SVM补检及可选条件成对复核。它是本地类别补检研究，不涉及联邦通信或聚合训练。

**原始数据集不上传。** 保留中文说明、数据来源链接、固定配置、小型划分/选样索引和历史结果。核心只用CPU，不需要GPU、PyTorch或预训练权重。此次发布只整理代码与文档并检查软件入口，没有新训练、预测或研究结果。

## 0. 先确认复现范围

| 内容 | 本仓库提供什么 |
| --- | --- |
| 普通树tree70、完整MAT-PV | 可运行核心代码、三队列配置、每队列五个固定种子 |
| TabM、ModernNCA、TabICL | 已有七配置对照结果和上游版本；不含它们的训练适配器及权重 |
| TON、RT、CIC39 | 固定输入文件清单、SHA256、划分元数据与核心入口 |
| CIC46 | 历史标签量诊断的来源说明；不含该诊断的训练入口 |

因此，本仓库可运行的核心比较是tree70对MAT-PV，不能将其称作全部七种配置的完整一键复现。

70是全部类别合计的“新增目标池标注”预算，并非整个训练过程只有70条标签。源标签、2,000条正常校准标签和100条保护类验证标签另计。两核心配置共享同一批70条；完整流程主树/记忆用前50条，二分类与成对SVM用全部70条。详见[核心方法与范围](docs/METHOD_AND_SCOPE.md)。

## 1. 下载代码并进入正确目录

在[本仓库](https://github.com/shengzu03/matpv-core-repro)点击Code → Download ZIP，解压到E盘，例如E:\MATPV_core。若已安装Git，也可选择下面的下载方式（二选一）：

~~~powershell
git clone https://github.com/shengzu03/matpv-core-repro.git E:\MATPV_core
~~~

本页所有后续命令都在**同时含README.md、configs、matpv_repro的项目根目录**执行，不是在它的上一层运行。

保留.gitattributes：该文件防止Windows Git自动转换换行。历史CSV与代码以原字节SHA256核验，转换换行会造成校验失败。

## 2. 建立Python环境，只检查软件

以下为Windows PowerShell示例，建议Python 3.12。所有环境、缓存及后续运行结果放在E盘；不要把数据下载进代码仓库。

~~~powershell
Set-Location E:\MATPV_core
New-Item -ItemType Directory -Force E:\MATPV_work\tmp,E:\MATPV_work\pip | Out-Null
$env:TEMP='E:\MATPV_work\tmp'
$env:TMP=$env:TEMP
$env:PIP_CACHE_DIR='E:\MATPV_work\pip'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
py -3.12 -m venv E:\MATPV_work\venv
E:\MATPV_work\venv\Scripts\python.exe -m pip install -r requirements.txt
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro doctor
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro check-package
E:\MATPV_work\venv\Scripts\python.exe tools/verify_core.py
~~~

doctor应无依赖导入错误；check-package应显示PACKAGE_HASHES_OK；verify_core应显示FROZEN_ALGORITHM_FUNCTIONS_OK。它们不训练模型。若未安装Python 3.12，先安装对应Python或自行指定已有的3.12解释器路径。

requirements.txt是兼容版本范围，不能冒称为全部历史实验的精确锁文件。发布环境与已留存的历史证据见[环境说明](docs/ENVIRONMENT.md)。不需要把项目打包安装为wheel。

## 3. 下载对应版本的数据，再核验原文件

按[数据来源及下载链接](docs/DATA_SOURCES.md)下载并解压。**必须匹配实际版本与SHA256**，文件名相同但内容不同不能代替。配置内source_files.relative_path相对于下面的数据根目录。

| dataset参数 | 数据根目录示例 | 需要的输入及重点类别 |
| --- | --- | --- |
| ton_iot | E:\IDS_data\TON | 指定镜像TON CSV；跨站脚本、中间人攻击 |
| rt_iot2022 | E:\IDS_data\RT | UCI的RT_IOT2022文件；慢速连接耗尽、ARP投毒 |
| ciciot2023 | E:\IDS_data\CIC39\data | Kaggle第1版固定34个CSV；跨站脚本、中间人ARP欺骗 |

RT不含跨站脚本标签。CIC39与官方46特征表示不可混用。TON必须保留本仓库metadata中的约1.1 MB冻结队列索引；它不含流量特征，但包含类别等元数据。

先从TON开始：

~~~powershell
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro init --dataset ton_iot --data-root E:\IDS_data\TON --run-root E:\MATPV_runs
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro check-data --dataset ton_iot --run-root E:\MATPV_runs
~~~

init只记录位置，check-data只核验文件。应看到RAW_SHA256_OK。出现missing或sha256_mismatches时先修正文件位置或下载版本，**不要跳过校验、改掉配置哈希或按成绩挑换数据文件**。

## 4. 准备固定输入，核对划分

~~~powershell
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro prepare --dataset ton_iot --run-root E:\MATPV_runs
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro verify-splits --dataset ton_iot --run-root E:\MATPV_runs
~~~

prepare扫描原始文件、执行固定预处理并生成五轮输入，不训练模型。verify-splits核对生成记录、分组和次序是否与历史元数据一致，应列出五个固定种子。此时不需要打开测试标签评价。

## 5. 先完整跑一对，再做五次配对

以下两条命令只训练并评价TON第一对，适合先检查自己的环境；它不是五轮成绩：

~~~powershell
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro run --dataset ton_iot --run-root E:\MATPV_runs --seed 2026103101
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro evaluate --dataset ton_iot --run-root E:\MATPV_runs --seed 2026103101
~~~

需要完整五轮时去掉seed。已经完成且文件校验通过的一对会跳过，其余继续执行：

~~~powershell
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro run --dataset ton_iot --run-root E:\MATPV_runs
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro evaluate --dataset ton_iot --run-root E:\MATPV_runs
~~~

只有run命令拟合模型；evaluate读取新运行的预测计算指标。评价前必须完成本次请求的全部预测并锁定其SHA256，缺少任何一臂时显示incomplete，不读测试标签。不要在不同阈值或标签方案间反复用测试成绩选优，不要用python -O关闭完整性断言。

## 6. RT与CIC39按同一顺序运行

每次选一个队列；以下变量只影响路径，不修改固定协议。RT先使用：

~~~powershell
$taskDataset='rt_iot2022'
$taskDataRoot='E:\IDS_data\RT'
~~~

轮到CIC39时改为：

~~~powershell
$taskDataset='ciciot2023'
$taskDataRoot='E:\IDS_data\CIC39\data'
~~~

然后对选中的队列依次执行：

~~~powershell
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro init --dataset $taskDataset --data-root $taskDataRoot --run-root E:\MATPV_runs
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro check-data --dataset $taskDataset --run-root E:\MATPV_runs
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro prepare --dataset $taskDataset --run-root E:\MATPV_runs
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro verify-splits --dataset $taskDataset --run-root E:\MATPV_runs
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro run --dataset $taskDataset --run-root E:\MATPV_runs
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro evaluate --dataset $taskDataset --run-root E:\MATPV_runs
~~~

RT与CIC39各自的五个种子均为2026110201至2026110205。若先跑单轮，run与evaluate都追加--seed 2026110201。不同队列输出到独立子目录，不把三队列的成绩合并为同一测试集。

## 7. 查看进度、结果和已有对照

~~~powershell
E:\MATPV_work\venv\Scripts\python.exe -m matpv_repro status --dataset ton_iot --run-root E:\MATPV_runs
~~~

运行终端逐阶段显示进度。status列出每轮是否准备、tree70/MAT-PV是否完成。新产物在E:\MATPV_runs\队列名：

| 文件或目录 | 内容 |
| --- | --- |
| config.json | 本次数据位置与固定科学配置 |
| environment_at_run.json | 本次真实运行环境，不能替代历史环境 |
| outputs/core_summary.json | 本次完成对数、均值、样本标准差及逐轮指标 |
| outputs/种子/core_results.json | 本次逐轮两种核心模型的结果 |
| outputs/core_predictions_frozen.json | 评价前锁定的预测SHA256 |

summary中的mean_percent、sample_sd_percent单位为百分数；rare_f1是指定两类F1的平均，不是所有类别宏F1。F1、正确类别检出率、查准率越高越好；正常误报率越低越好。

已有对照表位于reference/队列名/main_results.csv，数值单位以原表头为准，通常为0至1比例。它们是历史成绩，不能冒充同门新环境产生的成绩。新运行不会覆盖reference。

## 8. 常见错误与结果边界

- No module named matpv_repro：先进入包含本README、configs和matpv_repro的目录，再运行命令。
- 缺文件或SHA256不符：对照数据文档与config中的relative_path核验版本，不绕过检查。
- Run init first：dataset与run-root必须和此前init使用的一致。
- No prepared inputs或split mismatch：先完成prepare；若仍不符，保留错误日志，核对版本、环境及原始行顺序，不重新随机挑选划分。
- status有未完成项、evaluate显示incomplete：继续原目录的run，全部预测完成后再评价。失败重复也应保留。
- 与历史成绩有差别：先保留完整种子、环境版本及日志，核对记录/分组、标签查询与预测差异；不能挑选接近历史的轮次报告。

五轮划分相互有重叠，公开数据曾参与开发，是受控回顾性比较，不是独立部署盲测。TON初始索引构造规则未完整追回，因此保留实际冻结索引。条件成对复核在既有TON五轮中没有改变预测，不能将整体收益归功于它。保留RT未提升及CIC39绝对表现较低的记录。

发布前做过算法函数一致性、导入、命令入口及文件校验；**未在全新环境重新训练验证数值复现，也不承诺跨依赖版本逐位一致**。软件检查不等于新实验。完整参数与范围见docs/METHOD_AND_SCOPE.md，来源见provenance。

## 目录与许可

configs保存固定协议；matpv_repro保存核心代码；metadata与reference保存小型复现元数据及历史记录；docs保存数据、环境、方法和外部对照说明。原始数据、模型权重、运行产物和个人访问令牌不上传。

自有代码与原创说明采用[MIT许可证](LICENSE)，方便复制、修改和再分发，需保留版权及许可声明。数据集及其派生索引不因MIT获得重新授权，见[许可范围](LICENSE_STATUS.md)、[第三方说明](THIRD_PARTY_NOTICES.md)和[数据来源](docs/DATA_SOURCES.md)。论文尚未发表，不填写虚构DOI；引用代码时注明实际版本，见CITATION.cff。
