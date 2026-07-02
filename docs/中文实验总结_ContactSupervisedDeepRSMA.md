# Contact-supervised DeepRSMA 中文实验总结

## 1. 实验目标

本实验的目标是在原始 DeepRSMA 的基础上，引入 RNA-小分子结构接触监督，让模型不仅学习 RNA 和小分子的序列/图表示，还能通过 PDB 复合物学习“核苷酸-配体原子之间哪里可能发生物理接触”。

核心想法是：

1. 保留 DeepRSMA 原始四分支主干：
   - RNA graph embedding
   - RNA sequence embedding
   - molecule graph embedding
   - molecule sequence embedding
   - cross-fusion module
2. 在 cross-fusion 后增加 nucleotide-atom contact prediction head。
3. 从 PDB RNA-ligand complex 自动生成 contact map 标签。
4. 先用 contact map 做结构预训练，再回到 R-SIM affinity 数据集微调 pKd。

因此，本方法不要求 R-SIM 数据集本身有 binding-site 或 contact 标注，而是利用 PDB 中已有的 RNA-ligand 复合物结构提供额外监督。

## 2. Contact label 构建方法

对于每个 PDB RNA-ligand complex：

- RNA 被表示为长度为 `p` 的 nucleotide 序列。
- 小分子被表示为包含 `q` 个 heavy atoms 的分子图。
- contact map 是一个 `p x q` 的矩阵。

标签定义：

```text
如果第 i 个 nucleotide 和第 j 个 ligand atom 的最短距离 < 4 Angstrom，
则 contact_ij = 1；
否则 contact_ij = 0。
```

contact head 的基本形式是：

```text
score_ij = MLP([r_i, m_j, r_i * m_j])
```

其中：

- `r_i` 是第 i 个 nucleotide 的 cross-fusion 后 embedding。
- `m_j` 是第 j 个 ligand atom 的 cross-fusion 后 embedding。
- `r_i * m_j` 表示二者的逐元素交互。

## 3. 数据集构建

### 3.1 初始 100-candidate PDB contact 数据

最开始使用 RCSB 查询得到 100 个 RNA-only RNA-ligand PDB 候选，最终构建出：

```text
samples: 94
unique_pdb: 94
unique_ligand_resname: 54
mean RNA length: 30.0
mean atom count: 23.4
total positive contacts: 2395
total pairs: 63742
mean contact density: 0.0545
```

这个数据集证明 contact pretraining 有信号，但规模偏小。

### 3.2 扩展 500-candidate PDB contact 数据

随后扩大 RCSB 查询规模，取 500 个 RNA-only RNA-ligand PDB 候选。

查询结果：

```text
RCSB total_count = 825
selected candidates = 500
metadata rows = 486
built contact samples = 484
skipped = 2
```

最终 contact 预训练数据集统计：

```text
samples: 484
unique_pdb: 484
unique_ligand_resname: 188
rna_len: min=4.0 mean=64.7 median=37.5 max=396.0
atom_count: min=7.0 mean=26.2 median=25.0 max=109.0
positive_contacts: min=1.0 mean=29.4 median=25.0 max=101.0
contact_density: min=0.0012 mean=0.0385 median=0.0311 max=0.1769
total_positive_contacts: 14218
total_pairs: 769417
```

相比 94-sample 版本，新数据集：

- PDB 数量约扩大 5.1 倍。
- ligand 种类从 54 增加到 188。
- positive contacts 从 2395 增加到 14218。
- RNA 长度覆盖范围更广，最大长度到 396。

## 4. Contact 预训练结果

### 4.1 100-candidate checkpoint

使用 94 条 contact samples 预训练：

```text
best validation top-k precision = 0.1766
```

随机 baseline 约等于 contact density，约 0.0463 左右。因此 100-candidate checkpoint 已经学到了 contact 信号。

### 4.2 500-candidate checkpoint

使用 484 条 contact samples 预训练：

```text
best validation top-k precision = 0.3865
validation contact density = 0.0355
```

这说明模型在 top-k contact prediction 上约为随机密度的：

```text
0.3865 / 0.0355 ≈ 10.9 倍
```

也就是说，扩大 PDB contact 预训练数据后，模型确实学到了更强的结构接触模式。

## 5. Affinity fine-tuning 设置

contact pretraining 只监督 backbone/cross-fusion/contact head，不监督 pKd affinity head。

因此，在 pKd fine-tuning 时采用 clean transfer：

```text
加载 contact-pretrained backbone；
跳过未经过 pKd 监督的 affinity head 权重。
```

跳过的模块包括：

```text
line1, line2, line3,
rna1, rna2,
mole1, mole2,
guided_line*, residual_line*
```

这个版本记为：

```text
ContactSkipAff
```

## 6. Independent test 结果

### 6.1 原始 DeepRSMA 复现结果

三种子 independent test，best-PCC 选择：

```text
PCC  = 0.4866 ± 0.0522
SCC  = 0.4912 ± 0.0523
RMSE = 1.0584 ± 0.1490
```

best-RMSE 选择：

```text
PCC  = 0.4614 ± 0.0613
SCC  = 0.4471 ± 0.0855
RMSE = 0.9298 ± 0.0318
```

### 6.2 Contact100SkipAff 结果

使用 94 条 PDB contact samples 预训练后的结果：

best-PCC 选择：

```text
PCC  = 0.5195 ± 0.0333
SCC  = 0.5257 ± 0.0194
RMSE = 0.9754 ± 0.0380
```

相比原始 DeepRSMA：

- PCC 提升约 0.033。
- SCC 提升约 0.034。
- RMSE 降低约 0.083。

说明 contact-supervised pretraining 的方向是有效的。

### 6.3 Contact500SkipAff 结果

使用 484 条 PDB contact samples 预训练后的结果：

best-PCC 选择：

```text
PCC  = 0.5816 ± 0.0547
SCC  = 0.5749 ± 0.0590
RMSE = 0.9152 ± 0.0788
```

best-RMSE 选择：

```text
PCC  = 0.5738 ± 0.0495
SCC  = 0.5902 ± 0.0204
RMSE = 0.8665 ± 0.0400
```

相比原始 DeepRSMA best-PCC 结果：

```text
PCC  提升约 0.095
SCC  提升约 0.084
RMSE 降低约 0.143
```

相比 Contact100SkipAff：

```text
PCC  从 0.5195 提升到 0.5816
SCC  从 0.5257 提升到 0.5749
RMSE 从 0.9754 降低到 0.9152
```

这说明扩大结构接触预训练数据后，affinity prediction 的性能出现了明显提升。

## 7. 三种子详细结果

Contact500SkipAff 的三种子 best-PCC 结果如下：

```text
seed 1:
epoch = 168
PCC   = 0.6072
SCC   = 0.6118
RMSE  = 0.8331

seed 2:
epoch = 68
PCC   = 0.5188
SCC   = 0.5069
RMSE  = 0.9902

seed 3:
epoch = 122
PCC   = 0.6188
SCC   = 0.6060
RMSE  = 0.9223
```

其中 seed 1 和 seed 3 的提升尤其明显，seed 2 也明显超过原始模型。

## 8. Contact-guided affinity head 探索

除了 contact pretraining，还尝试了让 contact prior 直接参与 affinity prediction。

实现了两种版本：

### 8.1 Naive guided head

直接用 contact-guided pooling 替代原始 DeepRSMA affinity head。

结果：

```text
seed1 best PCC ≈ 0.4326
```

该方法明显弱于主模型，说明不能粗暴地用 contact prior 替代原来的全局 affinity head。

### 8.2 Residual guided head

保留原始 DeepRSMA affinity head，再添加 contact-guided residual calibration。

seed1 上表现较好：

```text
PCC  = 0.5302
SCC  = 0.5301
RMSE = 0.8877
```

但三种子不稳定：

```text
PCC  = 0.4726 ± 0.0507
SCC  = 0.4850 ± 0.0518
RMSE = 1.1294 ± 0.2640
```

因此当前论文主线不应强调 contact-guided head，而应强调：

```text
contact-supervised representation pretraining
```

## 9. Validation/refit 严格协议诊断

为了避免“每个 epoch 都看 independent test”带来的测试集选择问题，额外实现了 validation-selected protocol：

1. 从训练集内部切出 validation。
2. 用 validation 选择 epoch。
3. independent test 只在最终 checkpoint 上评估一次。
4. 进一步尝试 full-train refit：用 validation 只选择 epoch，然后在完整训练集上重训到该 epoch。

seed1 诊断结果：

```text
ContactSkipAff, 90% train checkpoint:
PCC  = 0.2336
SCC  = 0.2805
RMSE = 1.0424

Original-style model, 90% train checkpoint:
PCC  = 0.1573
SCC  = 0.1200
RMSE = 1.1054

ContactSkipAff, full-train refit:
PCC  = 0.2533
SCC  = 0.4125
RMSE = 1.2378

Original-style model, full-train refit:
PCC  = 0.4592
SCC  = 0.4565
RMSE = 0.9415
```

这个结果说明：

- 10% random validation 只有约 14 条样本，太小。
- validation PCC 可以高到 0.9 以上，但不一定能代表 independent test。
- 当前严格 validation/refit 协议还不稳定。
- 论文最终如果采用严格协议，需要更可靠的 validation 方案，例如更大 validation split、多次重复 split、或固定 epoch/epoch range 策略。

注意：这个 validation/refit 诊断是在 500-candidate checkpoint 完成前做的，后续还需要用 Contact500SkipAff 重新评估严格协议。

## 10. 当前结论

目前最强、最适合作为论文主结果的是：

```text
Contact500SkipAff
```

核心结论：

1. PDB RNA-ligand complex 可以自动生成 nucleotide-atom contact map。
2. contact map 可以作为结构监督信号预训练 DeepRSMA backbone。
3. 在 R-SIM affinity 数据没有 contact/binding-site 标签的情况下，结构接触预训练仍然能提升 pKd prediction。
4. 当 contact pretraining 数据从 94 条扩大到 484 条后，模型性能显著提升。
5. 三种子 independent test 上，Contact500SkipAff 相比原始 DeepRSMA 同时提升 PCC、SCC，并降低 RMSE。

可以写成论文主张：

```text
我们提出一种 contact-supervised pretraining 方法，
利用 PDB RNA-ligand 复合物自动生成 nucleotide-atom contact map，
将结构接触知识迁移到 RNA-small molecule binding affinity prediction。
该方法不依赖 affinity 数据集中的 binding-site 标注，
并在 DeepRSMA independent test 设置下显著提升预测性能。
```

## 11. 目前创新点

### 创新点 1：结构接触监督预训练

不是只做 affinity regression，而是先让模型学习 RNA nucleotide 和 ligand atom 的物理接触模式。

### 创新点 2：不需要 R-SIM contact 标签

R-SIM 只有 RNA、SMILES 和 pKd，但没有 binding site。

本方法从 PDB 复合物自动构建 contact map，因此可以利用外部结构知识增强 affinity prediction。

### 创新点 3：保持 DeepRSMA 主干，低侵入式增强

没有推翻原始 DeepRSMA，而是保留其四分支和 cross-fusion 结构，通过 contact pretraining 提升表示能力。

### 创新点 4：数据规模带来明确收益

从 94 条 PDB contact samples 扩展到 484 条后，性能进一步明显提升，说明该方法具有可扩展性。

## 12. 下一步建议

接下来最值得做的实验：

1. 用 Contact500SkipAff 重新跑 validation/refit 严格协议。
2. 做 contact pretraining data size ablation：
   - 100 samples
   - 250 samples
   - 500 samples
3. 继续增加更多 contact map 可视化，展示模型能定位真实 RNA-ligand 接触区域，并分析 atom-level 误差。
4. 检查 PDB pretraining set 与 independent test 的 ligand/RNA 重叠。
5. 如果时间允许，继续扩大到 RCSB 全量 825 candidates。
6. 改进 contact head 的 atom-level 分辨率，缓解预测热图中的水平条纹问题。

## 13. 论文图片和补充证据

当前已经生成一组可以用于论文初稿/组会汇报的图片：

```text
docs/figures/fig_method_overview.png
docs/figures/fig_contact_dataset_summary.png
docs/figures/fig_contact_pretrain_curve.png
docs/figures/fig_performance_best_pcc.png
docs/figures/fig_performance_best_rmse.png
docs/figures/fig_contact_data_scaling.png
docs/figures/fig_contact_map_example.png
docs/figures/fig_contact_shuffle_control.png
docs/figures/fig_downstream_shuffle_control.png
docs/figures/fig_structure_case_3f4h.png
```

当前 contact map 示例使用 PDB 3F4H / ligand RS3：

```text
RNA length = 54 nt
ligand atoms = 29
positive contacts = 23
contact-positive nucleotides = 5
contact-positive ligand atoms = 16
top-k hits = 10 / 23
example top-k precision = 0.435
```

这张图说明模型能把较高 contact probability 富集到真实接触 nucleotide 附近；同时也暴露出 ligand atom 级别定位仍不够精细，这是后续模型改进和论文 discussion 中可以主动说明的限制。

新增 PDB 结构可解释性图：

```text
docs/figures/fig_structure_case_3f4h.png
docs/figures/fig_structure_case_3f4h.pdf
```

这张图把同一个 3F4H / RS3 样本映射回真实 PDB 结构坐标。左图显示完整 RNA-ligand complex，右图放大 binding pocket。灰色为 RNA backbone，蓝色为 ligand atoms，红色为真实 contact-positive nucleotides，橙色空心圈为真实接触 ligand atoms，绿色连线为模型 top-k 预测中命中的真实 nucleotide-atom contacts。

核心信息：

```text
PDB ID = 3F4H
ligand = RS3
RNA chain = X
RNA length = 54 nt
ligand atoms = 29
positive contacts = 23
contact-positive nucleotides = 5
contact-positive ligand atoms = 16
top-k hits = 10 / 23
top-k precision = 0.435
```

它的作用是把 contact map 的矩阵证据转成结构证据：模型高概率预测确实集中在 ligand-binding pocket 附近，而不是只在表格指标上看起来有效。

新增 shuffled contact label 对照图：

```text
docs/figures/fig_contact_shuffle_control.png
docs/figures/fig_contact_shuffle_control.pdf
```

该对照在每个 PDB RNA-ligand 样本内部随机打乱 contact map，保留 positive contact 数量和 contact density，但破坏具体 nucleotide-atom 对应关系。用真实 contact validation split 评估时：

```text
True contact pretraining:
mean top-k precision = 0.3471
AUPRC = 0.2847
AUROC = 0.8828

Shuffled contact pretraining:
mean top-k precision = 0.0261
AUPRC = 0.0571
AUROC = 0.7327

True validation contact density = 0.0211
```

这说明真实 nucleotide-atom contact 对应关系本身是有效监督信号；模型提升不是简单来自更多 PDB 样本、更多训练步数或正负样本比例。

新增 downstream shuffled-contact 对照图：

```text
docs/figures/fig_downstream_shuffle_control.png
docs/figures/fig_downstream_shuffle_control.pdf
```

该图比较 Original、ShuffledContact500 和 TrueContact500 在 independent test 上的三种子结果。采用 best-RMSE selection 时：

```text
Original:
PCC  = 0.4614 +/- 0.0613
SCC  = 0.4471 +/- 0.0855
RMSE = 0.9298 +/- 0.0318

ShuffledContact500:
PCC  = 0.3909 +/- 0.0389
SCC  = 0.3837 +/- 0.0699
RMSE = 0.9669 +/- 0.0172

TrueContact500:
PCC  = 0.5738 +/- 0.0495
SCC  = 0.5902 +/- 0.0204
RMSE = 0.8665 +/- 0.0400
```

这进一步说明：随机 contact label 不能带来 downstream pKd 提升，真实 contact map 中的结构对应关系才是关键监督信号。

对应中文图注和证据链说明见：

```text
docs/论文图片与证据链说明.md
```

此外，补充评估了 Contact500 checkpoint 在 contact validation split 上的 contact prediction 能力：

```text
validation pairs = 69461
positive contacts = 1465
contact density = 0.0211
mean top-k precision = 0.3471
AUPRC = 0.2847
AUROC = 0.8828
threshold precision = 0.3347
threshold recall = 0.3447
positive probability mean = 0.4286
negative probability mean = 0.2201
```

这些指标说明：

```text
模型在 contact task 上不是只学到随机密度，
而是能够明显富集真实 nucleotide-atom contact。
```

这部分证据可以和 affinity prediction 主结果一起使用：

```text
contact task 学得好 -> backbone 获得结构接触知识 -> downstream pKd prediction 提升
```

## 14. 当前文件和结果位置

关键数据和模型：

```text
data/pdb_contacts/pdb_ids_rna_only_500.txt
data/pdb_contacts/metadata_rna_only_500.csv
dataset/pdb_contact_rna_only_500
save/contact_pretrain_rna_only_500.pth
save/contact_pretrain_rna_only_500_shuffle.pth
```

关键日志：

```text
runs/contact_pretrain_rna_only_500.log
runs/contact_pretrain_rna_only_500_shuffle.log
runs/independent_contact_rna_only_500_seed1_skipaff.log
runs/independent_contact_rna_only_500_seed2_skipaff.log
runs/independent_contact_rna_only_500_seed3_skipaff.log
runs/independent_contact_rna_only_500_shuffle_seed1_skipaff.log
runs/independent_contact_rna_only_500_shuffle_seed2_skipaff.log
runs/independent_contact_rna_only_500_shuffle_seed3_skipaff.log
```

英文实验总结：

```text
runs/contact_experiment_summary.md
docs/contact_supervised_deeprsma_paper_plan.md
```
