# Data and code availability for the archived JISA analyses

The core reproduction code and archived-analysis package are available at https://github.com/shengzu03/matpv-core-repro. Version v0.2.0 (November 11, 2026) is the fixed release for the additional prediction audits and hypothetical-cost analyses: https://github.com/shengzu03/matpv-core-repro/releases/tag/v0.2.0.

The directory `audits/jisa_20261110` contains row-level true-class labels, graph-stage and final predictions, alignment indices, and routing/transition codes for five repetitions of each main cohort (TON, RT, and CIC39), together with five separate CIC46 diagnostic repetitions. Five TON branch files additionally expose the archived memory and direct-SVM outputs. It also provides the frozen count tables, hypothetical-cost results, and bounded comparison with the closest literature. Portable verification and cost-analysis scripts are in `tools`; the Chinese audit guide explains how to verify these records without training a model or obtaining the raw datasets. The records support the analyses in main-text Table 9 and Supplementary Tables S18–S23.

The original traffic feature datasets are not redistributed. Their official sources, the actual mirrors and versions used, and file-verification procedures are documented in `docs/DATA_SOURCES.md`; third-party data and data-derived labels/indices remain subject to the applicable original release terms. The MIT license applies to the authors' code and original documentation, not to an assumed relicensing of third-party data.

The runnable model cores remain the tree70 baseline and full MAT-PV. This release does not provide every training adapter or every row-level prediction for all seven comparison configurations, model weights, complete diagnostic training pipelines, or a new independent validation dataset. The newly shared analyses use already completed runs and do not establish operational benefits or independent deployment performance.

中文说明：这是一份与v0.2.0公开范围对应的声明文本。本次未将其写入论文Word；原稿中“全部逐条预测仅存本地”的旧说法需在后续修订时按实际公开范围同步，不能把本次公开范围写成七种配置的全部逐条预测。
