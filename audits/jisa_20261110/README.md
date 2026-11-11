# JISA新增分析：已有预测审计与假设成本界限

发布版本v0.2.0（2026-11-11）；归档分析日期2026-11-10。本目录支持正文4.6、5.7、Table 9和补充材料S18–S23；同一协议与最近文献的比较对应相关工作2.4及Table S12。所有记录来自已完成实验，没有新增训练、推断、概率生成或阈值校准。

## 1. 先区分两个复现入口

- **只核对新增分析**：执行本页第2–4步。仅需Python 3.10及以上和标准库，不需要安装机器学习依赖，也不需要下载数据集。
- **从原始数据重新训练模型**：仓库根目录README的核心复现流程。这不是本目录核对命令的一部分；发布本版本时没有运行该流程。

新增公开的是预测输出、真实类别与行/组索引等派生研究记录，**不是原始流量特征数据**。记录包含类别标签，不能称为“完全没有数据”。不包含IP地址、数据集特征矩阵、模型权重、个人聊天、投稿信或未发表稿件。

## 2. 核验下载文件

下载完整仓库v0.2.0，在项目根目录执行：

```powershell
python tools/verify_audit_release.py
```

应输出`AUDIT_RELEASE_FILES_OK`。这是文件SHA-256校验，不运行模型。

## 3. 从已公开的逐条记录重新计数

```powershell
python tools/audit_archived_predictions.py --output ../MATPV_audit_check/prediction
```

输出目录必须尚不存在，以保护已有结果。程序读取20个压缩CSV：TON、RT、CIC39主队列各5轮，CIC46诊断队列5轮。CIC39与CIC46不是相同的实验队列。另读取5个TON分阶段CSV，核对图阶段、记忆、直接SVM和条件复核的逐条衔接。

程序输出`PUBLIC_PREDICTION_AUDIT_OK`，并核对以下内容与`expected/analysis_results.json`一致：逐轮A/B/C、净正确数A-B、新增目标FP=B+C、路由分区和召回上限、各类别来源、图阶段与最终指标，以及五轮均值和样本标准差。输出还包含15条TON顺序分支记录；顺序分解不是移除模块后的反事实消融。

各轮记录可能重叠。所有汇总先逐轮计算，再五轮等权平均，不把记录总和当作独立部署样本。

## 4. 从审计计数复算假设成本界限

```powershell
python tools/analyze_archived_cost.py --source ../MATPV_audit_check/prediction/analysis_results.json --output ../MATPV_audit_check/cost
python tools/verify_audit_release.py --prediction-output ../MATPV_audit_check/prediction/analysis_results.json --cost-output ../MATPV_audit_check/cost/cost_analysis.json
```

应输出`PREDICTION_OUTPUT_MATCHES_ARCHIVE`和`COST_OUTPUT_MATCHES_ARCHIVE`。成本推导脚本仅使用Python标准库，不训练模型。

令F=B+C、lambda=c_FN/c_FP且c_FP>0，则delta_L/c_FP=F-lambda*A。主要五轮平均界限为`sum(F_s/n_s)/sum(A_s/n_s)`，不是各轮F/A的算术平均，也不是`sum(F)/sum(A)`。主要界限：TON约1.66246、RT约0.46276、CIC39约2.45160；CIC46诊断约6.2，以归档精确分数为准。lambda等于界限时持平，严格大于时才表示该定义下平均损失下降。A=0的轮次另行处理。

这些成本是明确假设，未测量真实金额、响应损失或人工时间；不是部署建议。比较对象是**相同轮次图阶段预测到最终预测**，不是独立Tree70到完整MAT-PV。TON主结果中的Tree70召回4.93%不能替代这里的图阶段召回2.07%。

## 5. 文件与列的含义

| 路径 | 内容 |
| --- | --- |
| `predictions/*.csv.gz` | 20轮同序逐条真实类别、图阶段预测、最终预测、转移码、路由码及对齐索引 |
| `ton_stages/*.csv.gz` | TON5轮图→记忆→直接SVM→条件复核的既有整数输出 |
| `run_metadata.json` | 队列角色、种子、目标/保护类编号、类别名称、追加标签组成；类别名保留原记录拼写 |
| `tables/prediction` | 已归档的逐轮/队列账本及真实类别来源表 |
| `tables/cost_and_literature` | 已归档的成本界限、假设网格、顺序分支界限及文献差异表 |
| `expected` | 原始分析生成的冻结汇总，用于数值对照；没有重新选择有利轮次 |
| `literature/evidence.json` | 最近两篇文献的原始来源链接、证据位置和贡献边界 |
| `provenance` | 原归档的相对路径与SHA-256；历史原脚本另行保存作方法溯源 |

`predictions`每行有`row_position,test_id,test_group,test_hash,true_class,graph_class,final_class,transition_code,target_route_code`。ID、组号和输入哈希用于对齐，没有原始流量特征。`test_hash`保留原始字符串：TON归档使用十进制表示，部分其他队列使用十六进制表示。不要转换为电子表格浮点数。

`transition_code`：0未改标；1=A，错改对；2=B，对改错；3=C，错改另一错。`target_route_code`：0非目标记录；1真实目标判正常；2真实目标判保护类；3真实目标原已识别；4真实目标位于允许补认区域。

目标类为f，保护类为g，正常类编号为0。受限补认只允许非0/g/f的原攻击预测改为f；上限为(T0+K)/N。最终漏认由判正常、判保护类、允许区域内未恢复三部分组成。上限是给定预测与路由下的可覆盖范围，不保证当前模型能达到它。

## 6. 来源、许可与公开范围

数据原始来源和实际镜像链接见`../../docs/DATA_SOURCES.md`。本目录不重新授权原始数据及其类别/索引等派生材料，仍须遵守相应原发布方条款；原创分析代码和说明适用仓库MIT许可。图阶段硬标签由原归档概率和原阈值在2026-11-10确定性解码，本次只公开该结果并核对，不重新生成分数或阈值。

`provenance/original_scripts`保留历史源脚本原字节；它们含历史本机路径、依赖未分发的原始归档，不应直接运行。可移植复核入口是本页第2–4步，不需要那些本机路径。历史SHA记录是来源证据，不代表全部大型归档都已公开。

本版本公开的是这四个队列的MAT-PV图阶段/最终预测，以及TON已有分支输出；不包含全部七配置的逐条预测、全部训练适配器或全部诊断运行入口。有限新增标注、回顾性重叠划分、缺少最简组合对照与独立验证等证据边界保持不变。
