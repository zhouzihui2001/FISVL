# Fine-Grained Information Supplementation and Value-Guided Learning for Remote Sensing Image-Text Retrieval
本工作提出了一种细粒度信息补充与价值引导学习（fine-grained information supplementation and value-guided learning, FISVL）模型，通过融合推荐系统领域的先验知识进行特征增强，并采用价值导向的训练策略学习细粒度、高表达性与鲁棒性兼具的特征表示。具体而言，本工作设计了以下核心模块：
1. ​**细粒度信息补充模块（fine-grained information supplementation, FGIS）​**：通过融合全局与局部特征的视觉信息，增强模型对遥感图像多尺度特征的感知能力，解决传统方法中因特征粒度不足导致的表征局限性问题。
2. ​**双重损失优化机制**：针对模态内相似性过高的问题，提出加权对比损失（加权策略参考了样本重要性动态调整方法）与场景自适应细粒度感知损失，通过约束特征空间分布提升模型判别力。
3. ​**价值引导学习框架**：在训练阶段动态聚焦关键信息，通过自适应的注意力机制（类似场景自适应门单元设计）优化不同训练阶段的学习目标优先级，显著提升模型收敛效率。
本工作在RSICD与RSITMD数据集上验证了模型的有效性，结果表明本方法在细粒度特征学习与跨模态对齐任务中均达到领先水平。
![# FISVL框架图](assets/fine-grained_framework.png)
* 模型效果
  
![# FISVL在RSITMD和RSICD数据集上的效果](assets/FISVL_result1.png)
![# 可视化效果](assets/FISVL_result2.png)
* 引用
```bibtex
@article{zhou2024fine,
  title={Fine-Grained Information Supplementation and Value-Guided Learning for Remote Sensing Image-Text Retrieval},
  author={Zhou, Zihui and Feng, Yong and Qiu, Agen and Duan, Guofan and Zhou, Mingliang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
* 代码请见[FISVL-pytorch](FISVL-pytorch.zip)
