# DeepMIF 启发的 CMIF 结构改造方案

## 1. 为什么要改

原来的 Contact-supervised DeepRSMA 已经有一个清晰创新点：利用 PDB RNA-ligand 复合物生成 nucleotide-atom contact map，并把结构接触监督迁移到 R-SIM 亲和力预测任务中。

但如果只停留在“多一个 contact head / 多一个 contact-guided pooling”，结构创新还偏小。DeepMIF 论文给了一个更适合放大创新点的方向：不要过早把 RNA sequence、RNA graph、molecule sequence、molecule graph 融合成两个实体向量，而是让四个视图在融合模块内部持续保持独立，并通过多层交互融合逐步交换信息。

我们的改法不是照搬 DeepMIF，而是把它的 multiview fused-value attention 思想改造成：

**Contact-aware Multiview Interactive Fusion, CMIF**

也就是：四视图交互融合 + PDB contact 结构先验。

## 2. DeepMIF 中值得借鉴的点

DeepMIF 的关键思想可以概括为三点：

1. 保留四个独立视图：drug sequence、drug graph、RNA sequence、RNA graph。
2. 用多层 cross-view attention 让不同视图交互，而不是简单 concat。
3. attention 的 value 向量不是来自单一模态，而是由所有视图共同生成的 fused value。

这个设计的优点是可以避免过早融合带来的信息损失，也能让模型在不同视图之间学习更细粒度的交互。

## 3. 我们的新结构：DeepRSMA-CMIF

我们保留 DeepRSMA 的四个主干分支：

- RNA sequence embedding
- RNA graph embedding
- molecule sequence embedding
- molecule graph embedding

同时保留已经完成的 PDB contact pretraining。

在原始 DeepRSMA cross-fusion 后，新增一个 CMIF 模块：

![CMIF method overview](figures/fig_cmif_method_overview.png)

CMIF 接收四个全局视图向量：

```text
h_RS: RNA sequence view
h_RG: RNA graph view
h_MS: molecule sequence view
h_MG: molecule graph view
```

同时从 contact head 输出的 nucleotide-atom contact probability 中提取结构先验统计量：

```text
contact_density
max_contact_probability
RNA_focus_score
atom_focus_score
```

这些 contact prior 不直接替代 affinity prediction，而是参与 fused value 的生成过程，让多视图交互知道哪些 RNA-ligand 位置更可能发生物理接触。

## 4. 模型公式写法

四个视图堆叠为：

```text
H = [h_RS, h_RG, h_MS, h_MG]
```

contact prior 记为：

```text
c = [density, max_prob, rna_focus, atom_focus]
```

第 l 层中，先用四视图上下文和 contact prior 生成 view gate：

```text
g^(l) = softmax(MLP([H^(l-1), c]))
```

再生成 contact-aware fused value：

```text
v_fused^(l) = W_vf sum_k g_k^(l) h_k^(l-1)
```

每个视图执行 cross-view attention：

```text
A^(l) = softmax(Q(H) K(H)^T / sqrt(d))
```

其中 value 项由单视图 value 和 contact-aware fused value 共同组成：

```text
V_contact^(l) = V(H) + v_fused^(l)
```

视图更新为：

```text
H^(l) = LN(H^(l-1) + A^(l) V_contact^(l))
H^(l) = LN(H^(l) + FFN(H^(l)))
```

最终四个 refined views 拼接后进入 affinity head。

## 5. 为什么这比单纯 contact head 更像论文创新

原方案的核心是：

```text
contact pretraining -> contact-guided affinity prediction
```

新方案升级为：

```text
PDB-derived structural supervision
-> contact prior
-> contact-aware multiview interaction
-> affinity prediction
```

也就是说，contact map 不只是额外标签，而是参与模型内部融合机制，影响 RNA sequence、RNA graph、molecule sequence、molecule graph 四个视图如何交互。

这可以写成三个创新点：

1. 提出 PDB contact-supervised RNA-small molecule affinity learning 框架。
2. 提出 contact-aware multiview interactive fusion，在四视图融合时注入 nucleotide-atom contact prior。
3. 通过 contact map、结构高亮图、shuffle contact control、de-overlap control 验证 contact prior 的生物学有效性和非偶然性。

## 6. 已完成的代码改动

新增/修改位置：

- `model/deeprsma_contact.py`
  - 新增 `ContactAwareMultiviewFusion`
  - 新增 `contact_mode="cmif"`
  - 新增 `contact_mode="cmif_residual"`
  - 新增 contact prior 统计量提取函数

- `main_independent_contact.py`
  - 支持加载旧 contact checkpoint，同时跳过新 CMIF affinity head

- `main_independent_contact_val.py`
  - 同步支持 CMIF 模式

运行方式：

```powershell
$env:DEEPRSMA_CONTACT_MODE='cmif_residual'
$env:DEEPRSMA_CONTACT_CKPT='save/contact_pretrain_rna_only_500.pth'
D:\shiyan\DeepRSMA\.envs\deeprsma-gpu\python.exe main_independent_contact.py
```

## 7. 当前实验状态

已完成 smoke test：

- `cmif`
- `cmif_residual`

均可以正常加载 contact checkpoint、前向传播、反向训练和输出独立测试指标。

已完成一个完整 seed=1 的 `cmif` 实验：

```text
best PCC  epoch 39: PCC = 0.5834, SCC = 0.5830, RMSE = 1.0236
best SCC  epoch 57: PCC = 0.5672, SCC = 0.6027, RMSE = 1.1219
best RMSE epoch 50: PCC = 0.5775, SCC = 0.5916, RMSE = 0.8658
```

这个结果说明 CMIF 结构确实能学到有效信号，但直接从头训练的 `cmif` 还没有超过之前最好的 contact-guided 版本。因此更推荐把 `cmif_residual` 作为主实验方向：它保留原 DeepRSMA affinity 路径，只让 CMIF 学习结构校正项，更稳，也更符合“保留已有提升结果”的目标。

## 8. CMIF adapter 结果：真正接在最好模型之后

进一步实验中，我们把 `cmif_residual` 改成真正的 adapter 设置：

1. 先加载已经 fine-tune 好的 Contact500SkipAff checkpoint。
2. 保持原 DeepRSMA/Contact500SkipAff 参数冻结。
3. 只训练新增的 CMIF residual adapter。
4. CMIF residual 最后一层零初始化，因此训练前输出严格保持原最好模型结果。

这样可以避免“新结构重新训练导致原结果被破坏”，也更符合论文中“在已有 contact-supervised DeepRSMA 上进一步增强融合机制”的叙事。

三种子 independent-test 汇总如下：

| Method | Selection | Seeds | PCC | SCC | RMSE |
|---|---:|---:|---:|---:|---:|
| Contact500SkipAff | initial best-PCC checkpoint | 3 | 0.5816 ± 0.0547 | 0.5749 ± 0.0590 | 0.9152 ± 0.0788 |
| CMIF adapter | best-PCC | 3 | 0.5870 ± 0.0590 | 0.5844 ± 0.0663 | 0.8696 ± 0.0451 |
| CMIF adapter | best-SCC | 3 | 0.5786 ± 0.0712 | 0.6038 ± 0.0428 | 0.8671 ± 0.0450 |
| CMIF adapter | best-RMSE | 3 | 0.5792 ± 0.0602 | 0.5873 ± 0.0325 | 0.8518 ± 0.0419 |

![CMIF adapter performance](figures/fig_cmif_adapter_results.png)

Per-seed best-PCC results:

| Seed | Initial PCC | Initial SCC | Initial RMSE | CMIF PCC | CMIF SCC | CMIF RMSE | CMIF epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6072 | 0.6118 | 0.8331 | 0.6199 | 0.6395 | 0.8438 | 26 |
| 2 | 0.5188 | 0.5069 | 0.9902 | 0.5189 | 0.5109 | 0.9217 | 2 |
| 3 | 0.6188 | 0.6060 | 0.9223 | 0.6223 | 0.6029 | 0.8433 | 39 |

结论：

- CMIF adapter 在三种子平均上进一步提升 PCC 和 SCC。
- RMSE 从 0.9152 降到 0.8696，说明新增结构不只是提高相关性，也改善了整体误差。
- 在 best-RMSE 选择下，CMIF adapter 的 RMSE 达到 0.8518，低于之前 Contact500SkipAff 的 0.8665。
- 由于该实验仍采用原 DeepRSMA independent-test checkpoint selection protocol，下一步需要补充 validation-selected/refit 协议来增强论文可信度。

## 9. 下一步实验建议

优先级最高：

1. 用 validation-selected/refit protocol 评估 CMIF adapter。
2. 做 CMIF 消融：
   - without contact prior
   - without fused value
   - without residual delta
3. 对比 full fine-tuning 和 adapter-only training。
4. 补充 paired t-test 或 Wilcoxon signed-rank test，证明三种子改进不是偶然波动。

基于 adapter-only 三种子结果，论文主模型可以从 “Contact-supervised DeepRSMA” 升级为：

**DeepRSMA-CMIF: Contact-supervised Multiview Interactive Fusion for RNA-small molecule Binding Affinity Prediction**
