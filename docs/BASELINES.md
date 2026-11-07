# 外部对照版本

本核心包仅提供tree70和matpv运行入口。以下固定上游版本对应历史深度对照，保留以便定位，并未将完整训练适配器、源快照和权重一并分发。仅下载上游最新版并默认运行，不构成对本论文固定监督和阈值协议的复现。

| 对照 | 上游 | 固定提交 | 上游许可 |
| --- | --- | --- | --- |
| TabM | https://github.com/yandex-research/tabm | 28e47ae301c92ec37787dde1ce923a0793f405b4 | Apache-2.0 |
| ModernNCA 所在TALENT | https://github.com/LAMDA-Tabular/TALENT | 39b54024902955e51063a34bce17d514cd68ff42 | MIT |
| TabICL | https://github.com/soda-inria/tabicl | d83091b845a8c5f7c087bd54964f755a9e560f45 | BSD-3-Clause |

TabICL历史权重：tabicl-classifier-v2-20260212.ckpt，SHA256 bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0。核心入口不加载该权重。

历史共同协议和深度选轮规则保留在reference各队列protocol.json中。普通与均衡两个版本都保留，不能按测试表现只挑其中更有利的一个。
