# Contact-supervised DeepRSMA 正式论文推进判断

更新时间：2026-06-16

## 当前已经成立的结果

1. 复现原 DeepRSMA 代码并跑通 GPU 训练。

2. 在 DeepRSMA 主干上加入了 nucleotide-atom contact prediction head，保留原来的四路输入：
   RNA graph embedding、RNA sequence embedding、molecule graph embedding、molecule sequence embedding，以及 cross-fusion module。

3. 从 PDB RNA-ligand 复合物中构建了结构监督数据：
   nucleotide 与 ligand atom 距离小于 4 Angstrom 标记为 contact。

4. Contact pretraining 本身是有效的：
   - Contact500 true label：top-k precision 约 0.347，AUPRC 约 0.285，AUROC 约 0.883。
   - Shuffled-label control：top-k precision 约 0.026，AUPRC 约 0.057。
   - 去重 Contact500：440 个样本，top-k precision 约 0.382，AUPRC 约 0.368，AUROC 约 0.941。

5. 原论文式 reproduced protocol 下，Contact500 能提升 affinity prediction：
   - Contact500 best-PCC：PCC 0.5816 +/- 0.0547，SCC 0.5749 +/- 0.0590，RMSE 0.9152 +/- 0.0788。
   - Shuffled Contact500 downstream 明显更差：PCC 0.3909 +/- 0.0389，SCC 0.3837 +/- 0.0699，RMSE 0.9669 +/- 0.0172。

6. 已经补充了可解释性证据：
   - contact map 定性图；
   - shuffled-label 对照图；
   - PDB 结构级 qualitative case study；
   - validation protocol 对比图。

## 新增的严格性检查

### PDB-RSIM 重叠检查

脚本：`scripts/check_pdb_rsim_overlap.py`

发现：

| 检查项 | 结果 |
|---|---:|
| PDB contact samples | 484 |
| independent ligands | 48 |
| canonical SMILES exact matches | 23 |
| independent ligands with Tanimoto >= 0.90 | 8 |
| independent ligands with Tanimoto >= 0.80 | 9 |
| PDB RNAs with HIV-window identity >= 0.90 | 6 |
| PDB RNAs with HIV-window identity >= 0.80 | 8 |

解释：Contact500 中存在和 independent test 相似甚至相同的 ligand/RNA 片段，因此原论文式提升有潜在 overlap 风险。这个检查非常重要，正式投稿时必须报告。

### 去重 contact 数据集

脚本：`scripts/make_deoverlap_contact_dataset.py`

去重规则：

- ligand Tanimoto to independent ligands < 0.80；
- RNA window identity to independent HIV RNA < 0.80。

结果：

| 项目 | 数值 |
|---|---:|
| input samples | 484 |
| kept samples | 440 |
| excluded samples | 44 |
| excluded ligand Tanimoto >= 0.80 | 36 |
| excluded ligand exact match | 23 |
| excluded RNA identity >= 0.80 | 8 |

去重后 contact pretraining 仍然有效，说明模型确实学到了结构接触模式，而不是只靠记忆 independent test 中的相似样本。

## 下游 affinity 的严格结果

### 去重 checkpoint under reproduced protocol

使用去重 checkpoint 直接 fine-tune：

| Variant | Seed | Best PCC | SCC at best PCC | RMSE at best PCC |
|---|---:|---:|---:|---:|
| deoverlap plain init | 1 | 0.4439 | 0.4121 | 0.9885 |
| deoverlap residual/freeze20 | 1 | 0.4170 | 0.4724 | 1.1108 |
| deoverlap multitask w=1, steps=8 | 1 | 0.3838 | 0.3344 | 1.0883 |
| deoverlap multitask w=0.1, steps=2 | 1 | 0.3849 | 0.3385 | 1.0930 |

解释：严格去重后的 contact pretraining 保留了 contact prediction 能力，但直接迁移到 pKd affinity 的提升不稳定。简单 residual pooling 和固定权重 multi-task regularization 暂时没有改善。

### Validation-selected refit protocol

正式协议设置：

- 20% internal validation；
- validation RMSE 选择 epoch；
- 选定 epoch 后用 full train refit；
- independent test 只做最终评估。

Seed 1 结果：

| Method | Selected epoch | Test PCC | Test SCC | Test RMSE |
|---|---:|---:|---:|---:|
| Original val-selected | 183 | 0.4627 | 0.4664 | 0.9940 |
| Contact500 val-selected | 53 | 0.3419 | 0.3453 | 1.1451 |

图：`docs/figures/fig_validation_protocol_seed1.png`

解释：Contact500 在原论文式 test-selected protocol 下提升明显，但在更严格的 validation-selected/refit seed 1 下没有超过 original。这个结果说明当前还不能把 affinity 提升作为正式论文的唯一主张。

## 当前创新点

1. 用 PDB RNA-ligand 结构自动生成 nucleotide-atom contact map，作为 R-SIM affinity 数据之外的结构监督。

2. 在 DeepRSMA 主干上加入 auxiliary contact prediction head，使模型同时具备 affinity prediction 和 residue/atom-level interpretability。

3. 设计 shuffled contact label control，证明提升不是由额外训练轮数或噪声监督带来的。

4. 做了 PDB-RSIM overlap audit 和 de-overlapped contact pretraining，补上了正式生信/计算药物论文最容易被质疑的数据泄漏问题。

5. 生成 contact map 和 3D structure-level qualitative case study，使模型解释从矩阵层面走向结构层面。

## 目前是否足够正式投稿

我的判断：还不够投正式生信/计算药物主流期刊/会议。

原因不是 idea 不行，而是 affinity 主结果还没有在严格 protocol 下站住：

- contact prediction 本身很强；
- 原论文式 protocol 下 affinity 有提升；
- 但 validation-selected/refit seed 1 下 Contact500 没超过 original；
- 去重后 contact pretraining 能学结构，但 affinity transfer 不稳定。

现在更适合的定位是：

1. 作为一个很有潜力的结构监督方向；
2. 作为 workshop/short paper 的初步结果；
3. 或者继续补强后，发展成正式论文。

## 下一步最值得做的事

1. 改 validation 策略。

当前 random validation 和 independent HIV test 分布差异大。下一步应该做 scaffold-aware 或 independent-like validation，例如按 ligand scaffold、RNA family、RNA identity 分组，让 validation 更接近 blind RNA / blind molecule 场景。

2. 改 contact-to-affinity 融合方式。

现在的 residual/contact-weight pooling 太简单。更值得尝试：

- contact-aware interaction energy pooling；
- top-k sparse contact pooling；
- learnable contact query tokens；
- pairwise interaction MIL head；
- uncertainty-weighted multi-task loss；
- gradient surgery，避免 contact loss 压制 affinity loss。

3. 重新定义主结果。

如果 affinity 严格提升短期内不稳，可以把论文主线转成：

> Structure-supervised pretraining improves RNA-ligand contact localization and provides interpretable priors for affinity prediction.

然后 affinity improvement 作为 secondary result，而不是唯一卖点。

4. 补多 seed validation-selected。

至少需要：

- Original val-selected/refit seeds 1/2/3；
- Contact500 val-selected/refit seeds 1/2/3；
- Deoverlap contact val-selected/refit seeds 1/2/3；
- Shuffled contact val-selected/refit seeds 1/2/3。

5. 扩展外部证据。

正式计算药物论文最好再补：

- ligand scaffold split；
- blind RNA split；
- blind molecule split；
- external RNA-ligand affinity set，如果能找到；
- classical ML baseline，例如 Morgan fingerprint + XGBoost/RF；
- published baseline rerun或引用对比。

## 当前推荐结论

不要现在直接写成“我们的方法显著提升 affinity prediction”。更稳的写法是：

> 我们提出了 contact-supervised DeepRSMA，通过 PDB RNA-ligand 复合物自动构建 nucleotide-atom contact supervision，使 DeepRSMA 获得结构接触预测能力和更强的可解释性。Contact pretraining 在真实标签与 shuffled-label 对照中表现出显著差异，并在原论文复现协议下提升 affinity prediction。然而，在更严格的 validation-selected/refit protocol 下，affinity transfer 仍不稳定，提示下一步需要更强的 contact-to-affinity fusion 和更接近 blind test 的 validation strategy。

