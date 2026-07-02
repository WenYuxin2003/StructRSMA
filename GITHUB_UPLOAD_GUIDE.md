# GitHub Upload Guide for StructRSMA

This guide describes how to upload the StructRSMA code repository to GitHub for manuscript review and future release.

## 1. What Has Been Prepared

The repository now contains:

- `.gitignore`: excludes large datasets, checkpoints, logs, caches, and LaTeX build files.
- `README.md`: describes StructRSMA, its relationship to DeepRSMA, repository layout, data policy, and basic usage.
- `CONTACT_PRETRAINING.md`: documents contact-dataset construction and contact pretraining.

The following directories are intentionally excluded from git:

- `dataset/`
- `save/`
- `runs/`
- `data/pdb_contacts/`
- `data/representations_cv/`
- `data/representations_independent/`
- `data/RNA_contact/`
- `data/RSM_data/`
- `data/blind_test/`

Large files should be shared separately through Zenodo, Figshare, institutional storage, GitHub Releases, or another data repository.

## 2. Recommended Route: GitHub Desktop

Use this route if command-line `git` is not installed.

1. Install GitHub Desktop: https://desktop.github.com/
2. Open GitHub Desktop.
3. Choose `File -> Add local repository`.
4. Select this folder:

```text
D:\shiyan\DeepRSMA\DeepRSMA-master
```

5. If GitHub Desktop says this is not a repository, choose `create a repository`.
6. Repository name suggestion:

```text
StructRSMA
```

7. Keep it private before submission if you do not want the project to be public yet.
8. Check the changed-file list carefully. You should not see `dataset/`, `save/`, or large `.pth`/`.pt` files.
9. Commit with a message such as:

```text
Initial StructRSMA code release
```

10. Click `Publish repository`.

## 3. Command-Line Route

Install Git for Windows first: https://git-scm.com/download/win

Then open PowerShell:

```powershell
cd D:\shiyan\DeepRSMA\DeepRSMA-master
git init
git add .
git status
git commit -m "Initial StructRSMA code release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/StructRSMA.git
git push -u origin main
```

Before `git commit`, inspect `git status` carefully. If large files appear, stop and update `.gitignore` first.

## 4. Submission Advice

For a manuscript submission, avoid claiming that all data are included if raw data or checkpoints are not in the repository. A safer data/code statement is:

```text
The StructRSMA source code, preprocessing scripts, and manuscript-generation utilities are available at [GitHub URL]. Large processed tensors and trained checkpoints are excluded from git and will be deposited separately or made available upon reasonable request, subject to the licenses of the original R-SIM and PDB-derived resources.
```

If the journal requires permanent archival, create a Zenodo DOI after the GitHub repository is finalized.
