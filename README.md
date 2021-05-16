# GC3 - Group Communication with Context Codec for Lightweight Source Separation

This repository provides a minimalist's implementation of several GC3-based sequence modeling networks. ***GC3***, the abbreviation for ***G***roup ***C***ommunication with ***C***ontext ***C***odec, is a simple method to decrease both the model size and complexity of sequence modeling networks for source separation. Adding GC3 to existing models can maintain the same level of performance with only 4.7% model size (storage < 600KB on hard disk, Pytorch 1.4.0) and 17.6% MACs. For details of the design, please refer to our paper: [TASLP](https://ieeexplore.ieee.org/document/9427149) or [Arxiv](https://arxiv.org/abs/2012.07291).

To get an intuition about the effectiveness of GC3, we applied GC3 to four types of sequence modeling networks for source separation: DPRNN [[1]], TCN [[2]], UBlock [[3]], and Transformer [[4]]. The table below (Table V in the paper) provides the comparison of the separation performance, model complexity, memory footprint, and running speed of the four models and their GC3 counterparts:
![](https://github.com/yluo42/GC3/blob/main/GC3-stat.JPG)


[1]: https://ieeexplore.ieee.org/abstract/document/9054266/
[2]: https://ieeexplore.ieee.org/abstract/document/8707065/
[3]: https://ieeexplore.ieee.org/abstract/document/9231900/
[4]: https://arxiv.org/abs/2007.13975
