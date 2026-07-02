# StructRSMA：PDB 来源接触监督迁移学习用于 RNA-小分子结合亲和力预测

> 说明：这是当前 JCIM/ACS 英文 LaTeX 初稿 `structrsma_jcim.tex` 的中文阅读版，便于快速检查论文逻辑、方法叙事和实验结果。正式投稿仍以英文 LaTeX 稿为准。作者、单位、基金和数据仓库地址仍是占位内容，需要后续替换。

## 摘要

RNA-small molecule affinity datasets 通常只提供 pair-level affinity labels，这使得多视图模型很难直接学习局部 nucleotide-atom recognition patterns。为解决这一限制，本文提出 StructRSMA，一个建立在 DeepRSMA backbone 上的 contact-supervised transfer framework。StructRSMA 保留原始 DeepRSMA 的四个分支，即 RNA sequence、RNA graph、molecule sequence 和 molecule graph encoders，并将 PDB-derived nucleotide-atom contact supervision 作为辅助结构学习信号。

第一阶段，本文将 PDB RNA-ligand complexes 使用 4 Å 距离阈值转换为 binary contact maps，并训练 contact prediction head 学习局部 RNA-ligand contact patterns。第二阶段，contact-pretrained backbone 被迁移到 R-SIM affinity prediction；此时 R-SIM 不提供 contact labels。预训练好的 contact head 被复用，用于推断 R-SIM pair 的 contact probability maps；轻量级 Structural Contact Adapter（SCA）再总结 inferred contact prior，对 base affinity prediction 进行 residual calibration。

实验表明，contact pretraining 将复现的 DeepRSMA baseline 从 PCC 0.4866、RMSE 1.0584 提升到 PCC 0.5816、RMSE 0.9152。SCA 在 best-RMSE selection 下进一步将 RMSE 降低到 0.8518，同时保持相近的 correlation metrics。这些结果说明，PDB-derived contact supervision 可以为 RNA-small molecule affinity prediction 提供有用的可迁移结构先验。

关键词：RNA-small molecule interaction；binding affinity prediction；structural supervision；contact map；multiview learning；graph neural network；interpretability

## 1. 引言

RNA 参与转录、翻译、剪接、基因调控和催化等多种生物过程。RNA 折叠后形成的结构能够产生 pocket 和 surface，从而被 drug-like small molecules 识别。RNA 的治疗价值推动了 RNA 调控小分子的发现，包括作用于 riboswitch、viral RNA element、splicing regulator 和 disease-associated RNA structures 的化合物。然而，实验测定 RNA-ligand binding affinity 仍然成本高、周期长，因此在 RNA-targeted drug discovery 早期阶段，计算优先筛选具有重要价值。

早期 RNA-ligand recognition 计算研究主要使用 docking 和基于物理启发的 scoring functions，用于估计 binding pose 或 interaction energy。近期机器学习方法使用 molecular fingerprints、structural interaction fingerprints、target-specific models 和 RNA subtype-aware predictors 来建模 RNA-small molecule interactions。R-SIM 数据库进一步收集了 RNA-small molecule pairs 的定量 binding affinities，使数据驱动的 RSMA 建模成为可能。在此基础上，DeepRSMA 提出了 cross-fusion deep learning architecture，整合 RNA sequence、RNA graph、small-molecule sequence 和 small-molecule graph views，证明了细粒度序列和图信息能够改善 affinity prediction。DeepMIF 和其他多视图模型也进一步强调了 view-level interaction modeling 对 RSMA prediction 的重要性。

尽管已有方法取得进展，一个核心限制仍然存在：大多数 affinity datasets 只用每个 RNA-ligand pair 的全局 pKd 标签监督模型。这种弱监督要求模型从 pair-level regression signal 中间接推断局部物理识别模式。该问题有两个难点。第一，相比 protein-ligand 资源，可用 affinity 数据规模较小。第二，binding affinity 依赖稀疏的 nucleotide-atom interactions，而 loss function 只把整个 RNA-ligand pair 当作一个 scalar target。因此，模型可能能够较好预测 pKd，却没有真正学到 ligand 在 RNA 上接触的位置，从而限制了解释性和 structure-guided design 的下游应用价值。

PDB RNA-ligand complexes 提供了一种互补监督形式。虽然它们通常不总是和 quantitative affinity labels 配对，但它们包含 atomic coordinates，可以自动推导 nucleotide-atom contact maps。这些 contact maps 编码了哪些 RNA nucleotides 与哪些 ligand atoms 在空间上接近，因此能够为局部识别提供直接训练信号。已有 binding-site prediction 方法如 RLBind 表明 RNA-ligand contact information 对定位 binding regions 有用，但这种结构信号尚未在 DeepRSMA 框架中作为可迁移先验充分用于 affinity prediction。

StructRSMA 不是为了替换 DeepRSMA architecture，而是提出一个互补问题：实验解析的 RNA-ligand structures 是否能够为只有 pair-level labels 的 affinity prediction dataset 提供可迁移的 contact-level supervision？这个定位将本文贡献与单纯的 multiview fusion architecture 改进区分开来。本文目标是保留原始 DeepRSMA backbone，同时加入一个结构上有依据的 auxiliary task 和一个轻量 residual calibration module。

本文提出 StructRSMA，一种用于 RSMA prediction 的 PDB-derived contact-supervised transfer framework。StructRSMA 完整保留 DeepRSMA backbone，包括 RNA sequence embedding branch、RNA graph embedding branch、molecular graph embedding branch、molecular sequence embedding branch 和 cross-fusion module。在此基础上加入两个组件。第一，在 PDB-derived RNA-ligand contact maps 上训练 nucleotide-atom contact prediction head，使 backbone 在 affinity fine-tuning 前学习局部物理接触。第二，在 contact-supervised backbone 后接入 Structural Contact Adapter（SCA）。SCA 不替代 base affinity predictor，而是总结 inferred contact-prior statistics，并学习对 base affinity prediction 的 residual correction。该设计目标是保留已经验证有效的多视图主干，同时检验 contact supervision 是否能够从只有结构信息的 PDB complexes 迁移到只有 pair-level affinity labels 的 R-SIM prediction。

本文主要贡献如下：

1. 使用简单的 4 Å 距离规则，为 RNA-ligand complexes 构建 PDB-derived nucleotide-atom contact dataset，并评估 true-contact、shuffled-contact 和 de-overlapped contact settings。
2. 引入两阶段 contact-supervised transfer strategy：Contact500 用于预训练 shared backbone 和 contact head，R-SIM 在没有 contact labels 的情况下进行 affinity fine-tuning。
3. 提出 SCA，一种轻量级 contact-prior-guided residual calibration module，在不替换原 DeepRSMA backbone 的情况下校准 base affinity prediction。
4. 提供由 contact-map 和 3D structure visualization 组成的 qualitative evidence chain，将模型预测的高概率 contact 与真实 RNA-ligand geometry 连接起来。

## 2. 材料与方法

### 2.1 任务定义

给定 RNA sequence `R` 和 small molecule `M`，目标是预测 binding affinity `y`，本文使用 pKd 表示。对于 structural contact supervision，一个 RNA-ligand complex 被转换为 binary contact map：

```text
C ∈ {0,1}^{p × q}
```

其中 `p` 是 RNA nucleotides 的数量，`q` 是 ligand heavy atoms 的数量。若 nucleotide `i` 中任意 heavy atom 与 ligand heavy atom `j` 的最小距离低于 4 Å，则该 nucleotide-atom pair 被标记为 positive：

```text
C_ij = 1, if d(i,j) < 4 Å
C_ij = 0, otherwise
```

因此，contact task 是一个稀疏二值矩阵预测问题，而 affinity task 是一个标量回归问题。

### 2.2 StructRSMA 中保留的 DeepRSMA Backbone

StructRSMA 复用了完整的 DeepRSMA multiview backbone，而不是将其作为黑箱特征提取器。本文所说的 “preserved backbone” 指的是：原始 DeepRSMA 的 RNA feature extraction module、small-molecule feature extraction module、cross-fusion module 和 base affinity prediction path 都被保留。StructRSMA 不删除原始四个视图，而是在这些表示之上加入 contact supervision 和 SCA。

**RNA feature extraction module。** 原始 DeepRSMA 的 RNA 模块从两个互补视角表示 RNA：nucleotide sequence view 和 intramolecular graph view。在 graph view 中，RNA contact map 或 secondary-structure-derived map 被转换为 nucleotide graph。每个 nucleotide 是一个 node，edge 表示 RNA 内部 nucleotide 之间的结构接触关系。初始 node feature 来自 nucleotide embedding，然后使用 graph attention network（GAT）通过 attention-weighted aggregation 更新每个 nucleotide 的表示，得到：

```text
H_RG ∈ R^{p × d}
```

其中 `p` 是 RNA nucleotide 数量，`d` 是 hidden dimension。这个分支用于捕获线性序列中不明显的 RNA 结构邻域信息。

在 sequence view 中，DeepRSMA 将 trainable nucleotide embeddings 与 pretrained RNA-FM representations 结合。随后，nucleotide-level embeddings 经过多个不同 kernel size 的 1D CNN blocks，以提取局部和多尺度 sequence patterns，得到：

```text
H_RS ∈ R^{p × d}
```

也就是说，在进入 cross-fusion 前，RNA 侧同时拥有来自 sequence context 的 `H_RS` 和来自 graph-structured context 的 `H_RG`。RNA secondary-structure 和 contact features 可以由 ViennaRNA、bpRNA 等工具或资源生成；原始 DeepRSMA 实现中使用预测的 RNA contact information 作为 RNA graph construction 的一部分。

**Small-molecule feature extraction module。** 原始 DeepRSMA 的小分子模块同样保留两个视图。在 graph view 中，ligand SMILES 被转换为 molecular graph，其中 atoms 是 nodes，covalent bonds 是 edges。Atomic descriptors 作为 node features，随后使用 graph convolutional networks（GCNs）在 molecular topology 上传播信息，得到 atom-level graph representations：

```text
H_MG ∈ R^{q × d}
```

其中 `q` 是 ligand atom 数量。在 sequence view 中，SMILES string 被 tokenized，并由 transformer-style sequence encoder 编码。该 encoder 包含 multihead self-attention 和 feed-forward layers，用于学习 SMILES tokens 的上下文表示：

```text
H_MS ∈ R^{s × d}
```

其中 `s` 是 SMILES token length。因此，小分子侧的两个视图分别提供 atom-level topology information `H_MG` 和 SMILES-level sequence semantics `H_MS`。

**Cross-fusion module。** DeepRSMA 使用 transformer-based cross-fusion module 整合 RNA 和 ligand 的多视图信息。在 cross-attention 前，graph-view 和 sequence-view segment embeddings 会被加入对应 stream，使 transformer 能够区分 graph-derived tokens 和 sequence-derived tokens。RNA-side input 由 RNA graph stream 和 RNA sequence stream 组成，molecule-side input 由 molecule graph stream 和 SMILES sequence stream 组成。

Cross-fusion transformer 包含两个并行的 cross-attention 方向：RNA-to-molecule attention 使用 RNA tokens 作为 queries、ligand tokens 作为 keys/values；molecule-to-RNA attention 使用 ligand tokens 作为 queries、RNA tokens 作为 keys/values。因此，该 attention 机制能够建模 RNA 与小分子之间的 graph--graph、sequence--sequence 以及 graph--sequence cross-view pairs，而不是只在单一实体内部做 self-attention。

为了和图中符号保持一致，本文将 cross-fusion 后拆分得到的四条 streams 记为：

```text
H~_RS, H~_RG, H~_MS, H~_MG = CrossFusion(H_RS, H_RG, H_MS, H_MG)
```

这里每条 `H~` stream 都已经通过 cross-attention 融入了来自另一个 biomolecular entity 的信息。

**Base affinity prediction path。** StructRSMA 同样保留原始 DeepRSMA 的 affinity aggregation path。在这一路径中，四条 cross-fused streams 经过 masked mean pooling，得到四个 view-level vectors：

```text
h_RS = Pool(H~_RS)
h_RG = Pool(H~_RG)
h_MS = Pool(H~_MS)
h_MG = Pool(H~_MG)
```

这四个向量组成 multiview representation：

```text
H = [h_RS, h_RG, h_MS, h_MG]
```

这四个向量用于后续 SCA。对于保留的 base DeepRSMA predictor，RNA-side 和 molecule-side 信息会进一步与对应的原始 branch-level embeddings 结合，得到 pair-level vectors `h_R` 和 `h_M`，再拼接输入 affinity multilayer perceptron：

```text
y_base = MLP_aff([h_R, h_M])
```

这一路径构成 StructRSMA 的 primary affinity predictor，后续 SCA 只在其基础上学习 residual calibration。

**Token-level branch added for contact prediction。** 在保留 DeepRSMA backbone 的基础上，StructRSMA 额外加入 token-level contact branch。对于 contact prediction，模型保留 pooling 之前的 token-level states，因此图中的 `r_i` 和 `m_j` 不是额外冒出来的新特征，而是同一个 cross-fusion backbone 的 token-level outputs。具体来说，RNA sequence-view stream 和 RNA graph-view stream 在 nucleotide position 上是对齐的，因此 Nucleotide token fusion 模块将第 `i` 个 nucleotide 的表示定义为：

```text
r_i = 1/2 (H~_RS,i + H~_RG,i)
```

由此得到用于 contact prediction 的 nucleotide embeddings：

```text
r_1, r_2, ..., r_p
```

小分子侧需要更谨慎。Contact label 的列是 ligand atoms，而不是 SMILES tokens。SMILES token 与真实原子并不总是一一对应，所以 contact head 不直接使用 SMILES sequence token 作为 atom representation。Atom-level token selection 模块使用 cross-fused molecular graph stream 中第 `j` 个 atom 的表示：

```text
m_j = H~_MG,j
```

由此得到 ligand atom embeddings：

```text
m_1, m_2, ..., m_q
```

也就是说，SMILES sequence stream `H~_MS` 仍然参与 backbone、cross-fusion 和后续 pooled affinity/SCA branch；但在 nucleotide-atom contact prediction 这个分支里，它不作为 atom-indexed contact token 使用，因为 SMILES tokens 不能保证与 ligand atoms 一一对应。实现中 RNA 和 ligand graph streams 会 padding 到固定最大长度，并在 contact loss 与 contact-prior statistics 计算时使用对应 mask，避免 padding 位置影响训练和统计。

我们保留该 backbone，是因为原始 DeepRSMA 消融实验已经表明 graph views、sequence views 和 cross-fusion 都对最终预测有贡献。

图 1：StructRSMA 总体框架。StructRSMA 保留原始 DeepRSMA 四分支 backbone，并将 PDB-derived nucleotide-atom contact supervision 作为辅助结构预训练任务。在 Stage 1，PDB RNA-ligand complexes 被转换为 contact maps，用于训练 contact prediction head。在 Stage 2，R-SIM 只提供 pair-level pKd labels，因此不使用 contact labels。预训练好的 contact head 为 R-SIM pairs 推断 contact probability maps，SCA 总结 inferred contact prior，并预测对 base affinity output 的 residual correction。

### 2.3 PDB-Derived Contact Dataset

本文从 PDB structures 中提取包含 RNA polymer chains 和 nonpolymeric ligands 的 RNA-ligand complexes。过滤 water molecules、common ions、ambiguous ligands，以及不具备有效 RNA-ligand proximity 的样本。对于每个保留的 complex，解析 RNA nucleotide coordinates 和 ligand heavy-atom coordinates，并使用上述 4 Å 规则生成 nucleotide-atom contact map。每个 contact sample 包含 RNA sequence、ligand identity 或可用的 SMILES、ligand graph 和 binary contact map。

最终得到的 Contact500 set 包含 484 个可用 RNA-ligand contact samples。为了评估其与 downstream affinity test set 的潜在重叠，我们还生成 de-overlapped Contact500 subset：移除与 independent-test examples 具有高 ligand similarity 或高局部 RNA sequence identity 的 contact samples。de-overlapped subset 包含 440 个 contact samples。该 subset 作为额外控制，用于测试 contact signal 是否反映一般 structural learning，而不是记忆相似 RNA-ligand pairs。

这里需要明确区分 Contact500 和 R-SIM 在 StructRSMA 中的作用。Contact500 是本文新引入的结构监督数据集，它不是另一个 affinity dataset，也不提供 pKd 标签。PDB 中的三维坐标只用于生成 nucleotide-atom contact map。相比之下，R-SIM 提供的是 RNA-small molecule pair-level binding affinity labels，但没有 nucleotide-atom contact annotations。经过预处理后，两类数据都可以整理成模型可输入的 RNA-ligand 形式，并送入同一个 StructRSMA backbone；但它们监督的是不同的 prediction head。因此，从 PDB contact pretraining 到 R-SIM affinity prediction 的迁移，本质上迁移的是模型参数和结构归纳偏置，而不是直接迁移某个 PDB 样本的 embedding 或 contact label。

| Dataset | Source | Supervision label | Training role |
|---|---|---|---|
| Contact500 | PDB RNA-ligand complexes | Nucleotide-atom contact map `C` | 预训练 shared backbone 和 contact head |
| R-SIM | Measured RNA-small molecule affinities | Pair-level pKd value `y` | 微调 affinity head / SCA，用于 pKd prediction |

图 2：PDB-derived contact dataset 统计信息。Contact supervision 将 RNA-ligand structures 转换为 nucleotide-atom matrices，提供了 R-SIM 等 pair-level affinity datasets 中缺失的局部 interaction labels。

### 2.4 Contact Prediction Head 与 Loss

基于上面定义的 token-level 表示，Contact head 预测 cross-fused nucleotide embedding `r_i` 和 ligand atom embedding `m_j` 之间的 pairwise score：

```text
s_ij = MLP_contact([r_i, m_j, r_i ⊙ m_j])
```

其中 `⊙` 表示 elementwise multiplication。所有 nucleotide-atom pairs 的 scores 组成 contact logit matrix：

```text
S ∈ R^{p × q}
```

Contact probability matrix 为：

```text
P = sigmoid(S)
```

由于只有很少一部分 nucleotide-atom pairs 是 contacts，contact prediction 是高度类别不平衡任务。因此本文使用 focal loss：

```text
L_contact = - α(1 - p_t)^γ log(p_t)
```

其中 `α = 0.75`，`γ = 2.0`。该损失能够强调困难正样本，减少大量负样本对训练的支配。

### 2.5 两阶段接触监督迁移

StructRSMA 被设计为 contact-supervised transfer framework。

第一阶段，Contact500 只用于 structural contact pretraining。每个 PDB RNA-ligand complex 提供 binary nucleotide-atom contact map，但不使用 affinity label。Shared encoders、cross-fusion module 和 contact prediction head 通过 focal loss 优化：

```text
R, M → C
L_stage1 = L_contact
```

这一阶段的目标是让 backbone 学习局部 RNA-ligand recognition patterns，而不是学习全局 affinity calibration。

第二阶段，将 contact-pretrained parameters 迁移到 R-SIM affinity prediction。R-SIM 提供 pair-level pKd labels，但不提供 nucleotide-atom contact annotations。因此，在这一阶段 contact head 不再由 contact labels 监督，而是被复用来为每个 R-SIM RNA-ligand pair 推断 contact probability map，并将该 inferred map 总结为 SCA 使用的 contact-prior statistics。Affinity objective 为：

```text
R, M → y_hat
L_affinity = MSE(y_hat, y)
```

这一阶段需要强调：每一个 R-SIM RNA-ligand pair 都会用自己的 RNA 和 ligand 输入重新经过同一个 backbone 编码。模型不会直接复用某个 PDB complex 已经算好的 embedding。真正被复用的是第一阶段学到的 RNA encoder、molecule encoder 和 cross-fusion weights。到了 R-SIM 阶段，这些参数作为初始化继续训练，新的 token-level representations 会被 pooling 成 pair-level vectors，再用于 affinity regression。

在 DeepRSMA + Contact500 pretraining 设置中，载入 contact-pretrained backbone，同时重新初始化或重新训练 affinity head。这样可以避免过度依赖 contact-only pretraining 阶段并未充分训练的 affinity head，并让 R-SIM 负责校准全局 pKd。

### 2.6 Structural Contact Adapter

在 DeepRSMA + Contact500 pretraining 并完成 affinity fine-tuning 后，StructRSMA 进一步引入 SCA。SCA 被设计为保守的 residual calibration module，而不是 base DeepRSMA affinity predictor 的替代品。它的目的不是从零构建一个新的 global representation，而是判断 inferred contact prior 应该如何调整 base prediction。SCA 基于 contact-prior statistics 和四个 pooled view vectors 学习 residual correction：

```text
H = [h_RS, h_RG, h_MS, h_MG]
```

从预测的 contact probability matrix `P` 中，我们计算四个 contact summary statistics：

```text
c = [density, maxprob, rnafocus, atomfocus]
```

其中，density 表示整体预测 contact mass，maxprob 表示最强局部 contact，rnafocus 和 atomfocus 分别衡量预测 contact probability 在 RNA nucleotides 和 ligand atoms 上的集中程度。

SCA 首先计算 contact-prior-conditioned view gate：

```text
g = softmax(MLP_gate([H, c]))
```

得到 gated view representation：

```text
z_contact = W_c sum_{k=1}^{4} g_k h_k
```

与此同时，轻量级 attention 更新四个 view vectors：

```text
A = softmax(Q(H)K(H)^T / sqrt(d))
H' = A V(H)
```

最终 SCA residual 由 contact-guided vector、attended view summary 和 base affinity prediction 共同预测：

```text
Δy = MLP_sca([z_contact, pool(H'), y_base])
```

最终 StructRSMA prediction 为：

```text
y_StructRSMA = y_base + Δy
```

这种设计保持了改动的局部性：原始 backbone 仍然保留，而 SCA 学习结构接触证据应如何校准最终 affinity estimate。

### 2.7 实验设置与评价指标

Affinity prediction 使用 Pearson correlation coefficient（PCC）、Spearman correlation coefficient（SCC）和 root mean squared error（RMSE）评估。PCC 衡量预测 pKd 和实验 pKd 的线性一致性，SCC 衡量排序一致性，RMSE 衡量绝对回归误差。PCC 和 SCC 越高越好，RMSE 越低越好。

Contact prediction 使用 top-k precision、area under the precision-recall curve（AUPRC）和 area under the receiver operating characteristic curve（AUROC）评估。对于每个 complex，`k` 设置为 ground-truth positive contacts 的数量。Top-k precision 衡量预测 top-k pairs 中有多少是真实 contacts。由于 contact maps 非常稀疏，该指标适合评估模型是否能将真实局部 interactions 排到靠前位置。由于 nucleotide-atom contact maps 高度稀疏，本文主要关注 top-k precision 和 AUPRC；AUROC 作为次要指标报告，因为在强类别不平衡下它可能仍保持较高。

Downstream affinity experiments 报告三种 random seeds 的 mean 和 standard deviation。本文结果来自本地 independent-test reproduction protocol。由于 split protocol、training budget 和 checkpoint selection rules 与原始 DeepRSMA 和 DeepMIF 论文中的完整 five-fold cross-validation 结果不同，因此不应将这些数值直接等同于原论文正式 benchmark。

## 3. 结果与讨论

### 3.1 Contact Supervision 提供可学习的局部结构信号

第一个问题是 PDB-derived contact maps 是否包含可学习信号。由于 nucleotide-atom contact maps 高度稀疏，本文主要关注 top-k precision 和 AUPRC；AUROC 作为次要指标报告，因为它在强类别不平衡下可能仍然较高。在 Contact500 validation 上，true-contact model 达到 top-k precision 0.3471、AUPRC 0.2847 和 AUROC 0.8828。Shuffled-label control 保持相同 positive-contact density，但破坏 RNA-ligand geometry 与 labels 之间的对应关系，其 top-k precision 降至 0.0261，AUPRC 降至 0.0571。这一显著差距说明模型不是只利用 label sparsity 或矩阵形状，而是学习了结构化的 nucleotide-atom recognition patterns。

表 1：PDB-derived RNA-ligand contact maps 上的 contact prediction controls。

| Setting | Samples | Top-k precision | AUPRC | AUROC |
|---|---:|---:|---:|---:|
| True Contact500 | 484 | 0.3471 | 0.2847 | 0.8828 |
| Shuffled Contact500 | 484 | 0.0261 | 0.0571 | 0.7327 |
| De-overlap Contact500 | 440 | 0.3817 | 0.3678 | 0.9408 |

图 3：True contact supervision 与 shuffled-label control。保持 contact density 但打乱标签后，top-k precision 和 AUPRC 大幅下降，说明 contact labels 含有真实结构信息。

De-overlapped subset 达到 top-k precision 0.3817、AUPRC 0.3678 和 AUROC 0.9408。虽然 de-overlap subset 更小，但 contact metrics 仍然较强，说明 contact head 捕捉的是可迁移 structural patterns，而不仅依赖近重复样本。

### 3.2 Contact Pretraining 改善 Affinity Prediction

接下来，我们将 contact-pretrained backbone 迁移到 R-SIM affinity prediction。在本地 independent-test protocol 下，复现的 DeepRSMA baseline 在 best-PCC checkpoint selection 下达到 PCC 0.4866、SCC 0.4912 和 RMSE 1.0584。DeepRSMA + Contact100 pretraining 将 PCC 提高到 0.5195，并将 RMSE 降低到 0.9754。扩展到 DeepRSMA + Contact500 pretraining 后，PCC 进一步提高到 0.5816，RMSE 降低到 0.9152。该趋势支持以下假设：更丰富的 structural contact supervision 能够提供更强的 transferred affinity prior。

表 2：本地 independent-test protocol 中 contact-supervised variants 的 affinity prediction performance。数值为三种 random seeds 的 mean ± standard deviation。

| Method | PCC ↑ | SCC ↑ | RMSE ↓ |
|---|---:|---:|---:|
| Reproduced DeepRSMA | 0.4866 ± 0.0522 | 0.4912 ± 0.0523 | 1.0584 ± 0.1490 |
| DeepRSMA + Contact100 pretraining | 0.5195 ± 0.0333 | 0.5257 ± 0.0194 | 0.9754 ± 0.0380 |
| DeepRSMA + Contact500 pretraining | 0.5816 ± 0.0547 | 0.5749 ± 0.0590 | 0.9152 ± 0.0788 |

图 4：best-PCC checkpoint selection 下的 downstream affinity performance。Contact pretraining 改善复现的 DeepRSMA baseline，其中更大的 Contact500 set 在已测试的 contact-pretraining variants 中带来最强提升。

当 checkpoint selection 强调 RMSE 时，DeepRSMA + Contact500 pretraining 也能改善绝对误差。复现 baseline 的 RMSE 为 0.9298，而 Contact500 pretraining 将 RMSE 降到 0.8665。Shuffled-contact pretraining 未能复现同样收益，并且其 RMSE 差于 true-contact pretraining。该 downstream control 与 contact-task control 互相补充，说明 affinity gain 来自有意义的 contact supervision，而不仅仅是额外训练步骤。

图 5：Downstream shuffled-control experiment。True Contact500 pretraining 比 shuffled-contact pretraining 更稳定地改善 affinity prediction，说明 contact labels 的几何内容是有用的。

### 3.3 SCA 进一步校准 Contact-Supervised Backbone

由于 SCA 被设计为 residual calibration module，本文报告基于 PCC、SCC 和 RMSE 的多种 checkpoint-selection views，用于区分 ranking-oriented performance 和 absolute-error calibration。SCA 在 DeepRSMA + Contact500 pretraining model 之后评估。与 Contact500-pretrained best-PCC checkpoint 相比，加入 SCA 的 StructRSMA 将 PCC 从 0.5816 小幅提升到 0.5870，将 SCC 从 0.5749 提升到 0.5844，同时将 RMSE 从 0.9152 降低到 0.8696。在 best-SCC selection 下，StructRSMA 达到 SCC 0.6038 和 RMSE 0.8671。在 best-RMSE selection 下，StructRSMA 达到 RMSE 0.8518，是本地测试 variants 中最低的误差。

表 3：Contact500 pretraining 后 Structural Contact Adapter 的效果。数值为三种 random seeds 的 mean ± standard deviation。

| Method | Selection | PCC ↑ | SCC ↑ | RMSE ↓ |
|---|---|---:|---:|---:|
| DeepRSMA + Contact500 pretraining | initial best-PCC | 0.5816 ± 0.0547 | 0.5749 ± 0.0590 | 0.9152 ± 0.0788 |
| StructRSMA | best-PCC | 0.5870 ± 0.0590 | 0.5844 ± 0.0663 | 0.8696 ± 0.0451 |
| StructRSMA | best-SCC | 0.5786 ± 0.0712 | 0.6038 ± 0.0428 | 0.8671 ± 0.0450 |
| StructRSMA | best-RMSE | 0.5792 ± 0.0602 | 0.5873 ± 0.0325 | 0.8518 ± 0.0419 |

图 6：三种 seeds 上的 SCA 结果。Adapter 保留 contact-supervised backbone，并学习由 contact-prior statistics 和 four-view molecular representations 驱动的 residual calibration。

这些结果说明，PDB-derived contact supervision 改善了迁移后的 DeepRSMA representation，而 SCA 主要作为 calibration module 降低绝对预测误差。这对 DeepRSMA 的保守扩展是理想的：保留强原始 backbone，保留 contact-pretrained representation，并让 adapter 学习由结构接触证据驱动的小幅修正。RMSE 的改善尤其重要，因为在 virtual screening 场景中，绝对 pKd 尺度会影响 hit prioritization。

当前 SCA 有意使用 compact contact-prior statistics，而不是完整的 pairwise contact-map attention。这个设计选择反映了 Contact500 和 R-SIM 数据规模都较小。完整的 contact-map-conditioned attention module 会引入更多参数，并可能在小数据条件下过拟合。相比之下，本文使用的统计量提供了一种轻量方式，让 affinity predictor 能够接触 inferred structural evidence，同时保持原始 DeepRSMA representation 稳定。

### 3.4 定性案例研究：Contact Maps 与 3D Structural Evidence

为了检查预测 contacts 是否对应结构 interaction regions，我们可视化了 PDB 3F4H ligand RS3 的案例。该 RNA 有 54 个 nucleotides，ligand 有 29 个 heavy atoms。Ground-truth contact map 包含 23 个 positive nucleotide-atom contacts，涉及 5 个 contact-positive nucleotides 和 16 个 contact-positive ligand atoms。模型达到 top-k precision 0.435，即在预测 top 23 pairs 中有 10 个为真实 contacts。

图 7：PDB 3F4H ligand RS3 的 contact-map case study。左图为 ground-truth nucleotide-atom contacts，右图为 predicted contact probabilities，并标注 top-k true 和 false predictions。

Contact matrix 看起来稀疏且呈带状，这是因为只有少数 nucleotides 与 ligand 物理接近，并且相邻 ligand atoms 可能与同一个 nucleotide 具有相似距离。这种模式对 nucleotide-atom contact maps 是合理的：一个 ligand-binding pocket 通常会在少数 nucleotides 和化学相邻 ligand atoms 上形成正 contact cluster。因此，预测 heatmap 不应被理解为普通图像分割 mask；高概率行和列更适合作为 candidate interacting nucleotides 和 ligand atoms。

图 8：PDB 3F4H 的 3D structural case study。Contact-positive RNA nucleotides 和 ligand atoms 被高亮，用于将 matrix-level contact prediction 与真实 RNA-ligand geometry 连接起来。

将相同预测区域映射到 3D RNA-ligand structure 上，能够提供更直接的生物学解释。高亮 RNA nucleotides 在 ligand 周围形成局部 binding region，高概率 ligand atoms 对应于位于 RNA surface 附近的 atoms。这种可视化有助于连接 global affinity prediction 与 structure-guided interpretation，对 RNA-targeted design 中 prioritizing residues 或 ligand substituents 具有实用意义。

### 3.5 与已有 RSMA 模型比较

DeepRSMA 在 R-SIM 的 five-fold cross-validation 中报告了 state-of-the-art performance，优于 support vector machines、random forests、XGBoost 以及从 protein-ligand affinity prediction 迁移而来的 deep baselines。DeepMIF 进一步通过增强 multiview interaction modeling，并在融合前保持独立信息通道，提高了公开 benchmark 性能。

StructRSMA 的定位不同。它并不是用更大的 interaction module 替代 backbone，而是引入一种正交的监督来源：实验解析的 RNA-ligand structures。这使得本文贡献与 multiview architectural improvements 互补。原则上，contact pretraining 和 SCA 可以与未来更强的 backbone 结合，包括 DeepMIF-like multiview interaction modules，因为 contact task 作用于 nucleotide-atom level，并不依赖特定的 global affinity head。

### 3.6 局限性

本文仍有若干局限。第一，当前研究聚焦 controlled local reproduction，以隔离 contact supervision 的作用。未来工作应在 prior RSMA benchmarks 使用的 official five-fold cross-validation 和 blind-test protocols 下评估同一 contact-supervised transfer strategy。第二，PDB-derived contact set 规模仍然有限，而且 PDB structures 可能偏向研究较多、易结晶或易解析的 RNA motifs 和 ligands。第三，4 Å contact threshold 简单且可解释，但不能区分 hydrogen bonding、stacking、electrostatic contacts 或 water-mediated interactions。未来可以将 binary contact maps 扩展为 interaction-type-specific labels 或 distance distributions。第四，SCA 当前使用 predicted contact map 的 summary statistics；更丰富的 contact-conditioned attention over all nucleotide-atom pairs 可能进一步提升 representation learning，但在小数据集上需要谨慎正则化。

## 4. 结论

本文提出 StructRSMA，一种用于 RNA-small molecule binding affinity prediction 的 PDB-derived contact-supervised transfer framework。该方法保留原始 DeepRSMA four-view backbone 和 cross-fusion module，加入来自 RNA-ligand structures 的 nucleotide-atom contact pretraining，并使用 SCA 将 inferred contact priors 转化为 residual affinity calibration。True-contact 和 shuffled-contact controls 表明，PDB-derived contact maps 提供了可学习的结构信号。Downstream affinity experiments 表明，contact pretraining 改善了迁移后的 DeepRSMA representation，而 SCA 主要降低绝对预测误差。Contact-map 和 3D structure visualizations 将模型输出与局部 RNA-ligand geometry 连接起来。总体而言，这些结果支持将可用 RNA-ligand structures 转化为 transferable contact supervision，用于只有 pair-level labels 的 affinity prediction datasets。

## 数据和代码可用性

当前实现基于公开 DeepRSMA codebase 开发。R-SIM 可从其原始论文获取。PDB-derived contact dataset、preprocessing scripts、trained checkpoints 和 StructRSMA source code 将在数据清洗和文档整理完成后公开，或在论文接收后发布。当前稿件仍为 draft，其中作者和单位信息均为占位内容。

## Supporting Information

额外 training curves、validation-selection diagnostics、contact-data scaling experiments 和 de-overlap controls 已在本项目本地 supplementary figures 中生成。正式投稿前应整理独立 Supporting Information 文档。

## 致谢

作者感谢 DeepRSMA 开发者以及 R-SIM 和 PDB 资源维护者向社区开放数据和软件。基金信息需要在此处补充。

## 参考文献说明

正式参考文献见英文 LaTeX/BibTeX 文件：

- `docs/latex/structrsma_jcim.bib`
- `docs/latex/structrsma_jcim.bbl`

当前英文稿实际输出 28 条参考文献，覆盖 RNA 靶向小分子背景、R-SIM、DeepRSMA、DeepMIF、RNA-FM、GNN/Transformer 基础模型以及相关基线方法。
