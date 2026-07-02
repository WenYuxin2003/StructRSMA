# Contact-supervised DeepRSMA MVP

This branch keeps the original DeepRSMA backbone and adds an auxiliary
nucleotide-atom contact prediction head.

## Files

- `model/deeprsma_contact.py`: DeepRSMA backbone plus `ContactHead`.
- `data/pdb_contact_dataset.py`: loader and collate function for prebuilt contact samples.
- `scripts/query_rcsb_rna_ligands.py`: queries RCSB for RNA entries with nonpolymer ligands.
- `scripts/prepare_pdb_contacts.py`: downloads PDB files and infers a metadata CSV.
- `scripts/build_pdb_contact_dataset.py`: builds `.pt` samples from PDB RNA-ligand complexes.
- `scripts/summarize_contact_dataset.py`: reports contact pretraining dataset statistics.
- `pretrain_contact.py`: pretrains the contact head/backbone with contact supervision.

## Metadata format

Prepare a CSV such as `data/pdb_contacts/metadata.csv`:

```csv
pdb_id,pdb_file,smiles,rna_chain,ligand_resname,ligand_chain,ligand_resseq,embedding_file,ligand_atom_count,nearest_rna_distance
1fmn,1fmn.pdb,Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)O)c2cc1C,A,FMN,A,36,,31,2.591
```

Required:

- `pdb_file` or `pdb_id`: PDB filename under `--pdb-dir`.

Optional but strongly recommended:

- `smiles`: ligand SMILES. If omitted, the builder tries to derive it from the PDB ligand block.
- `rna_chain`
- `ligand_resname`
- `ligand_chain`
- `ligand_resseq`
- `embedding_file`: RNA-FM embedding `.npy` matching the RNA sequence length.

If no RNA-FM embedding is provided, the builder uses zero vectors with shape
`RNA length x 640` so the pipeline can run, but real pretraining should use
RNA-FM embeddings.

## Prepare PDB metadata

Query candidate PDB IDs from RCSB:

```powershell
D:\shiyan\DeepRSMA\.envs\deeprsma-gpu\python.exe scripts\query_rcsb_rna_ligands.py `
  --out data\pdb_contacts\pdb_ids.txt `
  --max-results 200
```

By default this keeps RNA-only entries, requires a nonpolymer ligand heavier
than 0.15 kDa, and limits polymer length to 512 residues. You can add
`--max-resolution 4.0 --experimental-method "X-RAY DIFFRACTION"` to build a
stricter crystallography-only set, or `--allow-protein` if you deliberately want
RNA-protein-ligand complexes.

For a first real-structure smoke test, download PDB files and infer the RNA
chain, ligand residue, ligand chain, ligand residue number, and ligand SMILES:

```powershell
D:\shiyan\DeepRSMA\.envs\deeprsma-gpu\python.exe scripts\prepare_pdb_contacts.py `
  --pdb-ids 1fmn 1uud `
  --pdb-dir data\pdb_contacts\pdb `
  --ccd-dir data\pdb_contacts\ccd `
  --metadata data\pdb_contacts\metadata_smoke.csv `
  --max-ligand-rna-distance 6.0
```

For batch use, put one PDB ID per line in a text file and pass
`--pdb-id-file pdb_ids.txt`.

## Build contact samples

```powershell
D:\shiyan\DeepRSMA\.envs\deeprsma-gpu\python.exe scripts\build_pdb_contact_dataset.py `
  --pdb-dir data\pdb_contacts\pdb `
  --metadata data\pdb_contacts\metadata.csv `
  --embedding-dir data\pdb_contacts\rnafm `
  --out-dir dataset\pdb_contact
```

The builder labels `contact[i, j] = 1` if nucleotide `i` has any heavy atom
within 4 Angstrom of ligand atom `j`.

Summarize the generated contact samples:

```powershell
D:\shiyan\DeepRSMA\.envs\deeprsma-gpu\python.exe scripts\summarize_contact_dataset.py `
  --data-dir dataset\pdb_contact
```

Important: atom-level labels require the PDB ligand atom order and molecule
graph atom order to match. The builder therefore builds the RDKit molecule from
the PDB ligand block and uses the SMILES molecule as a template to assign bond
orders.

## Contact pretraining

```powershell
D:\shiyan\DeepRSMA\.envs\deeprsma-gpu\python.exe pretrain_contact.py `
  --data-dir dataset\pdb_contact `
  --save-path save\contact_pretrain.pth `
  --epochs 30 `
  --batch-size 2 `
  --val-ratio 0.1 `
  --selection-metric topk_precision `
  --device cuda:0
```

The default loss is focal loss because nucleotide-atom contact maps are sparse.
The default focal-loss positive-class weight is `--alpha 0.75`, which is more
appropriate for sparse contact positives than the common object-detection
setting of 0.25.

## Affinity fine-tuning

After contact pretraining, fine-tune the same backbone on the R-SIM affinity
task by loading `save\contact_pretrain.pth` and optimizing the original pKd MSE
loss.

```powershell
$env:DEEPRSMA_CONTACT_CKPT='save\contact_pretrain.pth'
$env:DEEPRSMA_EPOCH='200'
$env:DEEPRSMA_BATCH_SIZE='8'
D:\shiyan\DeepRSMA\.envs\deeprsma-gpu\python.exe main_independent_contact.py
```

The fine-tuning script saves the best independent-test checkpoint to:

```text
save\model_independent_contact_<seed>.pth
```

Summarize independent-setting logs:

```powershell
D:\shiyan\DeepRSMA\.envs\deeprsma-gpu\python.exe scripts\summarize_independent_log.py `
  runs\independent_full_gpu.log `
  runs\independent_contact_rna_only_100_seed1.log
```
