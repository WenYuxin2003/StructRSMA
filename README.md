# StructRSMA

StructRSMA is a contact-supervised extension of DeepRSMA for RNA-small molecule binding affinity prediction. The method preserves the original DeepRSMA multiview backbone and adds PDB-derived nucleotide-atom contact pretraining plus a lightweight Structural Contact Adapter (SCA) for residual affinity calibration.

This repository is prepared for manuscript review and reproducibility. It contains the model code, preprocessing scripts, training entry points, manuscript figures/tables, and environment specification. Large raw datasets, processed tensors, and trained checkpoints are not tracked by git; see `.gitignore` and `CONTACT_PRETRAINING.md` for data preparation details.

## Relationship to DeepRSMA

The implementation was developed from the public DeepRSMA codebase:

DeepRSMA: a cross-fusion based deep learning method for RNA-small molecule binding affinity prediction.

Original authors: Zhijian Huang, Yucheng Wang, Song Chen, Yaw Sing Tan, Lei Deng, and Min Wu.

StructRSMA keeps the original RNA sequence, RNA graph, molecule sequence, molecule graph, cross-fusion, and affinity-prediction components, and adds the contact-supervised transfer components described in the manuscript.

## Repository Layout

- `model/`: DeepRSMA backbone and StructRSMA contact/SCA modules.
- `data/`: data loaders, tokenizers, and preprocessing code.
- `scripts/`: PDB contact-data construction, summary, plotting, and analysis scripts.
- `pretrain_contact.py`: contact pretraining entry point.
- `main_independent_contact.py`: independent-test affinity fine-tuning entry point with contact-pretrained initialization.
- `CONTACT_PRETRAINING.md`: detailed contact-dataset construction and training instructions.
- `docs/latex/`: manuscript source.
- `docs/figures/` and `docs/tables/`: generated manuscript figures and result tables.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate deeprsma
```

The exact environment name can be changed in `environment.yml`.

## Data

R-SIM affinity data should be obtained from the original R-SIM/DeepRSMA resources. PDB-derived RNA-ligand contact samples can be rebuilt with the scripts in `scripts/`. The contact construction workflow is documented in:

```text
CONTACT_PRETRAINING.md
```

Large files are intentionally excluded from git, including:

- raw and processed datasets under `dataset/`;
- downloaded PDB/mmCIF files under `data/pdb_contacts/`;
- RNA-FM/SPOT-RNA-derived representations;
- trained `.pth` checkpoints under `save/`;
- runtime logs under `runs/`.

## Basic Usage

Contact pretraining:

```bash
python pretrain_contact.py --data-dir dataset/pdb_contact_rna_only_500 --save-path save/contact_pretrain_rna_only_500.pth --device cuda:0
```

Affinity fine-tuning:

```bash
DEEPRSMA_CONTACT_CKPT=save/contact_pretrain_rna_only_500.pth python main_independent_contact.py
```

On Windows PowerShell:

```powershell
$env:DEEPRSMA_CONTACT_CKPT='save\contact_pretrain_rna_only_500.pth'
python main_independent_contact.py
```

## Manuscript Materials

The JCIM-style LaTeX manuscript source is in:

```text
docs/latex/structrsma_jcim.tex
```

The Overleaf-ready source directory is:

```text
docs/overleaf/structrsma_overleaf/
```

## Citation

If this code is used, please cite the StructRSMA manuscript after publication, as well as the original DeepRSMA work.
