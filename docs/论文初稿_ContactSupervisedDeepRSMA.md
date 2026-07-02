# Contact-supervised DeepRSMA: 面向可解释 RNA-小分子结合预测的结构接触监督预训练方法

> 中文初稿。本文档用于论文写作起草，实验结果以当前代码与日志为准。后续投稿前需要进一步补充英文润色、参考文献条目、统计显著性检验和多 seed 严格协议实验。

## 摘要

RNA-小分子相互作用预测是 RNA 靶向药物发现中的关键问题。已有深度学习方法通常直接从 RNA 序列、RNA 结构图、小分子序列和小分子图中学习全局表示，并回归结合亲和力。然而，亲和力数据集通常规模有限，且多数样本缺少 nucleotide-atom 级别的结合位点标注，导致模型虽然能够输出 pKd 等亲和力预测值，却难以解释具体的 RNA-小分子接触区域。为缓解这一问题，本文提出一种 contact-supervised DeepRSMA 框架，在保留 DeepRSMA 原有四分支主干和 cross-fusion module 的基础上，引入 nucleotide-atom contact prediction head，并利用 PDB RNA-ligand 复合物自动生成结构接触监督。对于每个 PDB 复合物，若 RNA nucleotide 与 ligand heavy atom 的距离小于 4 Angstrom，则将对应位置标记为 contact。模型先在 PDB-derived contact map 上进行结构预训练，再迁移到 R-SIM 亲和力预测任务。

实验表明，真实 contact supervision 能显著提升 contact prediction 任务表现。Contact500 模型在 validation split 上达到 top-k precision 0.3471、AUPRC 0.2847、AUROC 0.8828，显著高于 shuffled-label control；在去除与 independent test 相似的 PDB 样本后，de-overlapped contact 模型仍达到 top-k precision 0.3817、AUPRC 0.3678、AUROC 0.9408，说明模型学习到的结构接触模式并非简单的数据重叠记忆。在原 DeepRSMA independent-test 复现协议下，Contact500 pretraining 将 PCC 从 0.4866 提升到 0.5816，并将 RMSE 从 1.0584 降低到 0.9152。进一步的 shuffled-label control 表明，随机 contact 标签无法复现真实 contact pretraining 的下游收益。然而，在更严格的 validation-selected refit protocol 下，Contact500 在 seed 1 上尚未超过原始 DeepRSMA，提示当前结构接触监督向 affinity prediction 的迁移仍不稳定。总体而言，本文证明了 PDB-derived nucleotide-atom contact supervision 能够赋予 RNA-小分子模型更强的结构可解释性，并为后续构建更稳健的 RNA-ligand affinity predictor 提供了可扩展方向。

**关键词：** RNA-small molecule interaction；DeepRSMA；contact map；structure-supervised pretraining；binding affinity prediction；interpretability

## 1. 引言

RNA 靶点在抗病毒药物、抗菌药物、肿瘤治疗和基因调控相关治疗中具有重要价值。与蛋白质-小分子结合预测相比，RNA-小分子相互作用预测仍面临更严重的数据稀缺问题。一方面，具备定量亲和力标签的 RNA-ligand 数据集规模有限；另一方面，许多样本只有 RNA 序列、小分子 SMILES 和 pKd 等全局亲和力标签，缺少 nucleotide-level binding site 或 atom-level contact annotation。这使得纯监督 affinity prediction 容易受到数据规模、分布偏移和可解释性不足的限制。

DeepRSMA 是一个面向 RNA-小分子结合预测的多视图深度学习模型。它同时建模 RNA graph embedding、RNA sequence embedding、molecule graph embedding 和 molecule sequence embedding，并通过 cross-fusion module 学习 RNA 与小分子之间的交互表示。DeepRSMA 的消融实验表明，graph view、sequence view 和 cross-fusion module 均对最终性能有贡献。因此，本文不改变 DeepRSMA 的整体骨架，而是在其 cross-fusion 表示之后引入一个额外的 contact prediction head，使模型在亲和力预测之外同时学习 nucleotide-atom 级别的结构接触模式。

本文的核心思想是：虽然 R-SIM 等 affinity 数据集中缺少每条样本的 binding site label，但 PDB 中存在大量 RNA-ligand 复合物结构，可以通过几何距离自动生成 nucleotide-atom contact map。具体而言，对于 RNA 长度为 \(p\)、小分子 atom 数为 \(q\) 的复合物，我们生成一个 \(p \times q\) 的 binary contact map；当第 \(i\) 个 nucleotide 与第 \(j\) 个 ligand atom 的最近 heavy-atom 距离小于 4 Angstrom 时，对应位置记为 1，否则为 0。该 contact map 可作为辅助结构监督，使模型学习 RNA 与小分子在三维空间中发生物理接触的规律。

本文贡献如下：

1. 提出 contact-supervised DeepRSMA，在保留 DeepRSMA 原始四分支和 cross-fusion module 的基础上加入 nucleotide-atom contact prediction head。
2. 从 PDB RNA-ligand 复合物自动构建结构接触监督数据，无需 R-SIM affinity 数据集中提供 binding site annotation。
3. 通过 true-label、shuffled-label、de-overlap contact pretraining 和 downstream affinity fine-tuning 系统评估结构监督的作用。
4. 生成 contact map 和 PDB structure-level qualitative case study，将模型解释从全局 pKd 预测扩展到 nucleotide-atom interaction 层面。
5. 进一步引入 validation-selected refit protocol，诚实评估当前方法在更严格训练选择策略下的稳定性，并指出后续改进方向。

## 2. 材料与方法

### 2.1 任务定义

给定 RNA 序列及其结构图表示、小分子 SMILES 及其分子图表示，目标是预测 RNA-小分子结合亲和力 \(y\)，本文使用 pKd 作为回归标签。对于 PDB RNA-ligand 结构监督数据，额外定义 nucleotide-atom contact map：

\[
C \in \{0,1\}^{p \times q}
\]

其中 \(p\) 是 RNA nucleotide 数量，\(q\) 是 ligand heavy atom 数量。若 nucleotide \(i\) 与 ligand atom \(j\) 的最近 heavy-atom 距离小于 4 Angstrom，则 \(C_{ij}=1\)，否则 \(C_{ij}=0\)。

### 2.2 DeepRSMA 主干

本文保留 DeepRSMA 原始主干，包括四个输入分支：

- RNA graph embedding；
- RNA sequence embedding；
- molecule graph embedding；
- molecule sequence embedding。

四个分支输出经 cross-fusion module 得到 RNA 与 molecule 的交互表示，再通过 affinity head 输出 pKd。本文方法不删除任何原始分支，也不取消 cross-fusion module，以保持与 DeepRSMA 原始设计的一致性。

![Figure 1. Method overview](figures/fig_method_overview.png)

**图 1. 方法总览。** Contact-supervised DeepRSMA 保留 DeepRSMA 原有四路表征与 cross-fusion module，并在 cross-fusion token 表示之后增加 contact prediction head。模型先使用 PDB-derived nucleotide-atom contact map 进行结构预训练，再迁移到 R-SIM pKd prediction。

### 2.3 Contact prediction head

对于 cross-fusion 后的第 \(i\) 个 RNA nucleotide embedding \(r_i\) 和第 \(j\) 个 ligand atom embedding \(m_j\)，contact head 使用 pairwise MLP 计算 contact logit：

\[
s_{ij} = \text{MLP}([r_i, m_j, r_i \odot m_j])
\]

其中 \(\odot\) 表示 element-wise multiplication。所有 \(s_{ij}\) 组成 nucleotide-atom contact logit matrix。训练时对有效 nucleotide-atom pair 使用 binary cross entropy 或 focal loss。由于 contact map 中正样本极少，本文主要使用 focal loss：

\[
\mathcal{L}_{contact} =
- \alpha (1-p_t)^\gamma \log(p_t)
\]

其中 \(\alpha=0.75\)，\(\gamma=2.0\)。

### 2.4 PDB contact 数据构建

本文从 PDB RNA-ligand 复合物中生成 contact supervision。数据构建流程如下：

1. 收集含 RNA polymer 与非聚合小分子的 PDB 结构。
2. 过滤水、常见离子和不满足分子量/距离条件的 ligand。
3. 对 RNA nucleotide 与 ligand heavy atom 计算最近距离。
4. 若距离小于 4 Angstrom，则标记为 contact。
5. 将 RNA sequence、ligand molecule graph、SMILES 和 contact map 保存为 `.pt` 样本。

![Figure 2. Contact dataset summary](figures/fig_contact_dataset_summary.png)

**图 2. PDB contact 数据集统计。** 该图总结 PDB-derived contact dataset 中 RNA 长度、小分子 atom 数、positive contact 数量和 contact density 的分布。

### 2.5 两阶段训练策略

本文采用两阶段训练。

**阶段 1：结构接触预训练。** 使用 PDB contact dataset 训练模型预测 nucleotide-atom contact map：

\[
RNA + ligand \rightarrow C
\]

优化目标为：

\[
\mathcal{L}_{stage1} = \mathcal{L}_{contact}
\]

**阶段 2：亲和力微调。** 使用 R-SIM affinity dataset 训练模型预测 pKd：

\[
RNA + ligand \rightarrow pKd
\]

优化目标为：

\[
\mathcal{L}_{stage2} = \mathcal{L}_{affinity}
= \text{MSE}(\hat{y}, y)
\]

默认设置下，contact-pretrained backbone 被迁移到 affinity task，affinity head 从头训练。本文还探索了 residual contact fusion 和 contact regularization，但这些变体目前未优于简单迁移。

### 2.6 对照实验

本文设计以下对照，以验证 contact supervision 的有效性和稳健性。

**Shuffled contact control。** 在每个 PDB 样本内部随机打乱 contact map，保持 contact density 不变，但破坏 nucleotide-atom correspondence。该对照用于判断性能提升是否来自真实物理接触关系，而非额外预训练数据或训练轮数。

**De-overlap control。** 为降低 PDB contact dataset 与 R-SIM independent test 的潜在重叠风险，本文计算 PDB ligand 与 independent ligand 的 Morgan fingerprint Tanimoto similarity，并计算 PDB RNA 与 independent HIV RNA 的 window identity。去重规则为：移除 ligand Tanimoto \(\ge 0.80\) 或 RNA window identity \(\ge 0.80\) 的 PDB contact samples。

**Validation-selected refit protocol。** 原 DeepRSMA 复现协议会在 independent test 上每个 epoch 评估并报告最佳值。为评估更严格场景，本文额外使用 20% internal validation，以 validation RMSE 选择 epoch，再使用 full training set 重新训练到该 epoch，最后仅在 independent test 上评估一次。

## 3. 实验设置

### 3.1 数据集

**R-SIM affinity dataset。** 本文使用 DeepRSMA 原始 independent setting。训练集用于 pKd fine-tuning，independent test 包含 48 个 RNA-ligand pairs。20% validation-selected protocol 中，训练集被划分为 80% train 和 20% validation，其中 validation size 为 28。

**PDB Contact500 dataset。** Contact500 包含 484 个 PDB-derived RNA-ligand contact samples。每个样本包含 RNA sequence、ligand representation 和 nucleotide-atom contact map。

**De-overlap Contact500 dataset。** 在 Contact500 基础上移除与 independent test 相似的 ligand/RNA 样本后，保留 440 个 samples。

表 1 总结本文主要数据集。

| 数据集 | 样本数 | 标签 | 用途 |
|---|---:|---|---|
| R-SIM train | 140 | pKd | affinity fine-tuning |
| R-SIM independent test | 48 | pKd | independent evaluation |
| Contact500 | 484 | nucleotide-atom contact map | contact pretraining |
| De-overlap Contact500 | 440 | nucleotide-atom contact map | leakage-aware contact pretraining |

**表 1. 主要实验数据集。**

### 3.2 评价指标

Affinity prediction 使用：

- PCC：Pearson correlation coefficient；
- SCC：Spearman correlation coefficient；
- RMSE：root mean squared error。

Contact prediction 使用：

- top-k precision：对每个样本取预测概率最高的 \(k\) 个 pair，其中 \(k\) 等于真实 positive contact 数；
- AUPRC：area under precision-recall curve；
- AUROC：area under ROC curve；
- threshold precision/recall：使用 probability threshold 0.5 的二分类指标；
- contact density：positive contacts 占全部有效 nucleotide-atom pairs 的比例。

## 4. 结果

### 4.1 Contact pretraining 能学习真实 nucleotide-atom 接触关系

图 3 展示 Contact500 pretraining 过程中 train/validation loss 和 top-k precision 的变化。随着训练进行，validation top-k precision 明显高于 contact density baseline，说明模型能够从 PDB structure-derived contact labels 中学习到非随机的结构接触模式。

![Figure 3. Contact pretraining curve](figures/fig_contact_pretrain_curve.png)

**图 3. Contact pretraining 训练曲线。** 模型在 PDB-derived contact map 上进行结构预训练，validation top-k precision 随训练提升，并明显高于 contact density baseline。

表 2 汇总 contact prediction 结果。真实 Contact500 明显优于 shuffled-label control。进一步地，de-overlap Contact500 在移除与 independent test 相似样本后仍取得更高 AUPRC 和 AUROC，说明 contact pretraining 并非依赖简单数据重叠。

| Contact pretraining | Top-k precision | AUPRC | AUROC | Contact density |
|---|---:|---:|---:|---:|
| True Contact500 | 0.3471 | 0.2847 | 0.8828 | 0.0211 |
| Shuffled Contact500 | 0.0261 | 0.0571 | 0.7327 | 0.0211 |
| De-overlap Contact500 | **0.3817** | **0.3678** | **0.9408** | 0.0175 |

**表 2. Contact prediction 结果。** Top-k precision、AUPRC 和 AUROC 均表明真实 contact supervision 显著优于 shuffled-label control。

### 4.2 Shuffled-label control 证明真实 contact label 是有效信号

图 4 比较真实 contact labels 与 shuffled contact labels 在 contact task 上的表现。Shuffled labels 保持每个样本的 contact density，但破坏 nucleotide-atom 对应关系。结果显示 shuffled model 的 top-k precision 和 AUPRC 均明显下降，说明模型并非只学习 contact density 或样本统计偏差。

![Figure 4. Contact shuffled-label control](figures/fig_contact_shuffle_control.png)

**图 4. Contact task 的 shuffled-label control。** 真实 contact labels 显著优于 shuffled labels，说明 nucleotide-atom correspondence 是关键监督信号。

### 4.3 原论文式 independent-test 协议下 contact pretraining 提升 pKd prediction

在 DeepRSMA 原论文式 independent-test 复现协议下，Contact500 pretraining 相比 Original DeepRSMA 在 PCC、SCC 和 RMSE 上均有改善。表 3 展示 3 seeds 的 mean ± standard deviation。

| 方法 | Selection | PCC | SCC | RMSE |
|---|---|---:|---:|---:|
| Original DeepRSMA | best PCC | 0.4866 ± 0.0522 | 0.4912 ± 0.0523 | 1.0584 ± 0.1490 |
| Contact500 | best PCC | **0.5816 ± 0.0547** | **0.5749 ± 0.0590** | **0.9152 ± 0.0788** |
| Original DeepRSMA | best RMSE | 0.4614 ± 0.0613 | 0.4471 ± 0.0855 | 0.9298 ± 0.0318 |
| Contact500 | best RMSE | **0.5738 ± 0.0495** | **0.5902 ± 0.0204** | **0.8665 ± 0.0400** |

**表 3. 原论文式 independent-test 协议下的 affinity prediction 结果。**

图 5 和图 6 分别展示 best-PCC 和 best-RMSE selection 下的 performance comparison。

![Figure 5. Independent-test performance, best PCC](figures/fig_performance_best_pcc.png)

**图 5. Independent-test affinity prediction performance under best-PCC selection。**

![Figure 6. Independent-test performance, best RMSE](figures/fig_performance_best_rmse.png)

**图 6. Independent-test affinity prediction performance under best-RMSE selection。**

### 4.4 Contact data scaling 显示结构监督规模对下游结果有影响

本文比较不同规模 contact pretraining 数据对 downstream affinity prediction 的影响。随着 PDB contact 样本数从 0 增至 100 和 500，原论文式 independent-test protocol 下的 affinity performance 总体改善，说明更大规模结构监督可能有助于 RNA-ligand 表征学习。

![Figure 7. Contact data scaling](figures/fig_contact_data_scaling.png)

**图 7. Contact pretraining 数据规模效应。** Contact sample 数量增加后，downstream affinity prediction 在原论文式协议下呈改善趋势。

### 4.5 Downstream shuffled control 表明真实 contact supervision 优于随机结构标签

为了排除“额外 PDB 预训练”本身造成提升的可能，本文比较 true Contact500 与 shuffled Contact500 在 downstream pKd prediction 中的表现。结果显示 shuffled Contact500 无法复现 true Contact500 的提升。

| 方法 | Selection | PCC | SCC | RMSE |
|---|---|---:|---:|---:|
| Shuffled Contact500 | best PCC | 0.3932 ± 0.0371 | 0.3763 ± 0.0737 | 3.2104 ± 2.0374 |
| Shuffled Contact500 | best RMSE | 0.3909 ± 0.0389 | 0.3837 ± 0.0699 | 0.9669 ± 0.0172 |
| True Contact500 | best RMSE | **0.5738 ± 0.0495** | **0.5902 ± 0.0204** | **0.8665 ± 0.0400** |

**表 4. Downstream shuffled-contact control。** 随机 contact labels 无法带来与真实 contact labels 相同的 affinity gain。

![Figure 8. Downstream shuffled-label control](figures/fig_downstream_shuffle_control.png)

**图 8. Downstream shuffled-contact control on independent-test affinity prediction。**

### 4.6 Contact map 可视化显示模型能定位真实接触区域

图 9 展示 PDB 3F4H ligand RS3 的 contact prediction case。左图是真实 nucleotide-atom contact map，右图是模型预测的 contact probability。该样本包含 54 个 nucleotide 和 29 个 ligand atoms，共 23 个 positive contacts。模型 top-k prediction 命中 10 个真实 contact，top-k precision 为 0.435。

![Figure 9. Contact map example](figures/fig_contact_map_example.png)

**图 9. PDB 3F4H ligand RS3 的 contact map example。** 真实 contact map 与预测 contact probability 对比显示，模型能在稀疏 contact map 中将部分真实 nucleotide-atom contacts 排到较高概率区域。

### 4.7 Structure-level case study 提供更直观的生物结构解释

矩阵形式的 contact map 能展示 nucleotide-atom pair 的预测结果，但对生物结构直观性有限。因此本文进一步将 contact prediction 映射回 PDB coordinate space，并绘制 RNA-ligand complex 的 structure-level qualitative case。

![Figure 10. Structure case study](figures/fig_structure_case_3f4h.png)

**图 10. PDB 3F4H 的 structure-level visualization。** RNA backbone、ligand atoms、真实 contact-positive nucleotides 和模型 top-k true contact predictions 被映射到结构空间中，用于展示 contact-pretrained model 是否能聚焦 binding pocket 附近的真实接触区域。

### 4.8 PDB-RSIM overlap audit 与去重实验

为了评估 PDB contact pretraining 是否可能与 R-SIM independent test 存在重叠，本文进行了 overlap audit。结果显示 Contact500 中存在一定数量与 independent test 相似或相同的 ligand/RNA 片段。因此本文构建 de-overlap Contact500，以 ligand Tanimoto < 0.80 且 RNA window identity < 0.80 为保留条件。

| 检查项 | 结果 |
|---|---:|
| Contact samples | 484 |
| Independent ligands | 48 |
| Canonical SMILES exact matches | 23 |
| Independent ligands with Tanimoto >= 0.90 | 8 |
| Independent ligands with Tanimoto >= 0.80 | 9 |
| PDB RNAs with HIV-window identity >= 0.90 | 6 |
| PDB RNAs with HIV-window identity >= 0.80 | 8 |

**表 5. PDB-RSIM overlap audit。**

De-overlap 数据集构建结果如下：

| 项目 | 数值 |
|---|---:|
| Input contact samples | 484 |
| Kept samples | 440 |
| Excluded samples | 44 |
| Excluded ligand Tanimoto >= 0.80 | 36 |
| Excluded ligand exact match | 23 |
| Excluded RNA identity >= 0.80 | 8 |
| Total positive contacts kept | 12843 |

**表 6. De-overlap Contact500 数据集。**

去重后 contact prediction 仍然表现良好，但 affinity transfer 下降，说明 contact-to-affinity transfer 仍需进一步改进。

| Variant | Seed | Best PCC | SCC at best PCC | RMSE at best PCC |
|---|---:|---:|---:|---:|
| De-overlap plain init | 1 | 0.4439 | 0.4121 | 0.9885 |
| De-overlap residual/freeze20 | 1 | 0.4170 | 0.4724 | 1.1108 |
| De-overlap multitask w=1, steps=8 | 1 | 0.3838 | 0.3344 | 1.0883 |
| De-overlap multitask w=0.1, steps=2 | 1 | 0.3849 | 0.3385 | 1.0930 |

**表 7. De-overlap contact pretraining 的 downstream affinity transfer。**

### 4.9 严格 validation-selected refit protocol 暴露 affinity transfer 不稳定

原论文式 independent-test protocol 能展示与 DeepRSMA 复现结果的一致比较，但每个 epoch 评估 independent test 并报告最佳值，存在 test-set selection 风险。本文因此补充 validation-selected refit protocol：使用 20% internal validation，以 validation RMSE 选择 epoch，再用 full train refit 到选定 epoch，并最终评估 independent test。

| 方法 | Selected epoch | Test PCC | Test SCC | Test RMSE |
|---|---:|---:|---:|---:|
| Original val-selected | 183 | **0.4627** | **0.4664** | **0.9940** |
| Contact500 val-selected | 53 | 0.3419 | 0.3453 | 1.1451 |

**表 8. Validation-selected refit protocol under seed 1。**

![Figure 11. Validation-selected refit protocol](figures/fig_validation_protocol_seed1.png)

**图 11. Validation-selected refit protocol 下的 independent-test 结果。** 在 seed 1 中，Contact500 未能超过 Original DeepRSMA，提示当前 contact pretraining 对 affinity prediction 的迁移尚不稳定。

## 5. 讨论

本文的实验结果表明，PDB-derived nucleotide-atom contact supervision 是一个有效的结构学习信号。真实 contact labels 在 contact prediction 任务上显著优于 shuffled labels，并且在去除与 independent test 相似样本后仍保持较强性能。这说明模型并非仅依赖 contact density、样本统计偏差或数据重叠，而是确实学习到一定程度的 RNA-ligand physical contact pattern。

在 affinity prediction 方面，Contact500 在原 DeepRSMA independent-test 复现协议下取得明显提升，且 shuffled-contact control 无法复现该提升。这支持了 contact supervision 对 RNA-ligand 表征学习存在潜在帮助。然而，de-overlap downstream transfer 和 validation-selected refit protocol 暴露出两个关键问题。第一，PDB contact dataset 与 R-SIM independent test 存在一定分布或结构相似性，去除高相似样本后 affinity gain 明显减弱。第二，当不再使用 independent test 进行 epoch selection，而改用 internal validation 选择模型时，Contact500 在 seed 1 下未超过 Original DeepRSMA。这说明当前方法在 affinity generalization 上仍不够稳定。

这并不否定 contact-supervised pretraining 的价值，而是提示 contact-to-affinity transfer 需要更强的模型设计。当前实现主要将 contact-pretrained backbone 用作初始化，或通过简单 contact-weighted pooling、residual head 和固定权重 contact regularization 进行融合。这些方式可能无法充分表达 binding energy 中复杂的局部相互作用，也可能在小规模 pKd fine-tuning 时发生结构知识遗忘。未来更值得探索的方向包括 contact-aware interaction energy pooling、top-k sparse contact pooling、multiple-instance learning over nucleotide-atom pairs、uncertainty-weighted multi-task learning、gradient conflict mitigation，以及 scaffold-aware 或 RNA-family-aware validation split。

从论文定位上，当前最稳妥的主张不是“本文已经显著解决 RNA-ligand affinity prediction”，而是“本文提出了一种结构接触监督预训练框架，显著提升 nucleotide-atom contact localization，并为 RNA-ligand affinity prediction 提供可解释的结构先验”。Affinity improvement 可作为在原论文复现协议下观察到的 positive result，同时严格 validation-selected result 应作为 limitation 诚实报告。

## 6. 结论

本文提出 contact-supervised DeepRSMA，通过 PDB RNA-ligand 结构自动生成 nucleotide-atom contact map，并将其作为辅助结构监督引入 DeepRSMA 主干。实验结果显示，模型能有效学习真实 nucleotide-atom contact，并在 shuffled-label control 与 de-overlap contact evaluation 中保持显著优势。Contact map 和 structure-level visualization 进一步表明，该方法能够提供比全局 pKd 更细粒度的结构解释。在原论文式 independent-test 复现协议下，Contact500 pretraining 改善了 affinity prediction；但在更严格的 validation-selected refit protocol 下，该提升尚未稳定。因此，本文为 RNA-小分子结合预测提供了一条具有可解释性的结构监督预训练路线，同时也指出了未来提升 affinity generalization 的关键挑战。

## 7. 局限性

1. 当前 PDB contact supervision 与 R-SIM affinity dataset 之间存在一定 overlap risk，虽然本文进行了 de-overlap audit，但仍需要更大规模、更系统的外部验证。
2. Validation-selected refit protocol 目前只完成 seed 1，尚需扩展到 3-5 seeds。
3. 当前 contact-to-affinity fusion 较简单，未能在严格 protocol 下稳定提升 pKd prediction。
4. 本文尚未完成 scaffold split、blind RNA split、blind molecule split 和 external affinity dataset evaluation。
5. 参考文献、统计显著性检验和英文论文润色仍需补充。

## 8. 后续工作

为了将本文推进到正式生信/计算药物方向投稿，建议优先完成以下工作：

1. 补齐 validation-selected refit protocol 的 3 seeds：Original、Contact500、De-overlap Contact500、Shuffled Contact500。
2. 构建更接近 independent test 的 validation split，例如 ligand scaffold-aware split 或 RNA-family-aware split。
3. 改进 contact-to-affinity fusion，例如 top-k contact MIL pooling 或 contact-aware interaction energy head。
4. 扩展 PDB contact dataset，报告 ligand diversity、RNA family diversity 和 contact density distribution。
5. 加入 classical ML baselines，例如 Morgan fingerprint + XGBoost/RF，以及可运行的 RNA-ligand baseline。
6. 将本文翻译成英文并补齐参考文献。

## 图表清单

| 编号 | 文件 | 说明 |
|---|---|---|
| Figure 1 | `figures/fig_method_overview.png` | 方法总览 |
| Figure 2 | `figures/fig_contact_dataset_summary.png` | Contact dataset 统计 |
| Figure 3 | `figures/fig_contact_pretrain_curve.png` | Contact pretraining 曲线 |
| Figure 4 | `figures/fig_contact_shuffle_control.png` | Contact shuffled-label control |
| Figure 5 | `figures/fig_performance_best_pcc.png` | Best-PCC affinity performance |
| Figure 6 | `figures/fig_performance_best_rmse.png` | Best-RMSE affinity performance |
| Figure 7 | `figures/fig_contact_data_scaling.png` | Contact data scaling |
| Figure 8 | `figures/fig_downstream_shuffle_control.png` | Downstream shuffled control |
| Figure 9 | `figures/fig_contact_map_example.png` | Contact map example |
| Figure 10 | `figures/fig_structure_case_3f4h.png` | Structure-level case study |
| Figure 11 | `figures/fig_validation_protocol_seed1.png` | Validation-selected refit protocol |

## 参考文献占位

[1] DeepRSMA 原始论文。投稿前需要补充完整题名、作者、期刊、年份和 DOI。

[2] DeepDTA / DeepDTAF 相关药物-靶点亲和力预测方法。投稿前需要补充完整引用。

[3] GraphDTA 及图神经网络药物-靶点亲和力预测方法。投稿前需要补充完整引用。

[4] Transformer 和 cross-attention 相关基础方法。投稿前需要补充完整引用。

[5] RNA-ligand structure and binding database / PDB 相关资源。投稿前需要补充完整引用。

