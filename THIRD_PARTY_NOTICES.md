# 第三方方法、软件与数据说明

## 自有实现与方法借鉴

MIT仅适用于本仓库自有代码和原创使用说明，不把已有树模型、图残差传播、高斯分类或支持向量机算法重新宣称为原创。历史实现来源与移植改动记录在provenance/source_manifest.json。

代码通过正常导入使用NumPy、SciPy、scikit-learn、pandas与threadpoolctl；未将这些项目的实现或二进制依赖打包。安装者获得的依赖仍适用各自许可证与版权声明：

| 依赖 | 上游许可文件 |
| --- | --- |
| NumPy | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| SciPy | https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| scikit-learn | https://github.com/scikit-learn/scikit-learn/blob/main/COPYING |
| pandas | https://github.com/pandas-dev/pandas/blob/main/LICENSE |
| threadpoolctl | https://github.com/joblib/threadpoolctl/blob/master/LICENSE |

TabM、ModernNCA和TabICL仅保留历史实验记录与上游来源，不随本仓库分发其训练适配器、源快照或预训练权重。上游地址和历史版本见docs/BASELINES.md。

## 数据与派生复现材料

原始数据不在本仓库内。TON、RT与CIC的实际下载版本、SHA256、原始发布者和镜像区别，见docs/DATA_SOURCES.md与configs。数据镜像的标记不扩大原始发布者授予的权利。

metadata/ton_iot/flow_input_group_index.npz包含原行号、输入哈希、分组和类别type，不含流量特征。reference中的splits、query50、query20等包含重建同一划分所需元数据。这些数据派生材料不因根目录MIT许可证被重新授予数据使用权。

TON官方来源与用途条件：https://research.unsw.edu.au/projects/toniot-datasets 。RT官方许可与引用信息：https://archive.ics.uci.edu/dataset/942/rt-iot2022 。CIC官方来源：https://www.unb.ca/cic/datasets/iotdataset-2023.html 。具体实际镜像及限制以数据来源文档为准。

v0.2.0的audits/jisa_20261110新增由既有预测产生的研究输出、真实类别与对齐索引。数据来源和许可范围与上述队列一致，不上传原始流量特征或前人论文全文。literature目录仅保存作者撰写的比较与公开来源链接。
