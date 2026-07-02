import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .feature import atom_to_feature_vector, bond_to_feature_vector
from .process_data_molecule import WordVocab, atom_dict, max_len, smilebet


def load_torch(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def encode_smiles(smiles, vocab_path="data/smiles_vocab.pkl"):
    drug_vocab = WordVocab.load_vocab(vocab_path)
    content = []
    flag = 0
    while flag < len(smiles):
        if flag + 1 < len(smiles) and smiles[flag:flag + 2] in drug_vocab.stoi:
            content.append(drug_vocab.stoi[smiles[flag:flag + 2]])
            flag += 2
        else:
            content.append(drug_vocab.stoi.get(smiles[flag], drug_vocab.unk_index))
            flag += 1

    if len(content) > max_len:
        content = content[:max_len]

    mask = torch.ones(max_len, dtype=torch.float32)
    mask[len(content):max_len] = 0

    padded = list(content)
    if max_len > len(padded):
        padded.extend([drug_vocab.pad_index] * (max_len - len(padded)))

    atom_token_positions = []
    for i, token_id in enumerate(padded):
        if token_id in atom_dict:
            atom_token_positions.append(i)

    smiles_f = torch.from_numpy(smilebet.encode(smiles.encode("utf-8").upper())).long()
    return torch.tensor(padded, dtype=torch.long), torch.tensor([len(content)], dtype=torch.long), mask, atom_token_positions, smiles_f


def molecule_to_data(mol, smiles, y=0.0, entry_id=-1, vocab_path="data/smiles_vocab.pkl"):
    atom_features = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    atom_features = atom_features[:128]
    x = torch.tensor(np.array(atom_features, dtype=np.int64), dtype=torch.long)

    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i >= 128 or j >= 128:
            continue
        feature = bond_to_feature_vector(bond)
        edges.append((i, j))
        edge_features.append(feature)
        edges.append((j, i))
        edge_features.append(feature)

    if edges:
        edge_index = torch.tensor(np.array(edges, dtype=np.int64).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features, dtype=np.int64), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)

    smile_emb, smile_len, mask, atom_len, smiles_f = encode_smiles(smiles, vocab_path=vocab_path)

    data = Data()
    data.x = x
    data.graph_len = len(x)
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.smiles_ori = smiles
    data.y = torch.tensor([float(y)], dtype=torch.float32)
    data.e_id = torch.tensor([int(entry_id)], dtype=torch.long)
    data.smile_len = smile_len
    data.mask = mask
    data.smile_emb = smile_emb
    data.atom_len = atom_len
    data.smiles_f = smiles_f
    return data


def rna_to_data(sequence, edge_index, emb=None, y=0.0, entry_id=-1, target_id="pdb"):
    mapping = {"A": 0, "U": 1, "G": 2, "C": 3, "T": 1, "X": 4, "Y": 5}
    encoded = [[mapping.get(base.upper(), 4)] for base in sequence]
    x = torch.tensor(encoded, dtype=torch.float32)
    if emb is None:
        emb = torch.zeros((len(sequence), 640), dtype=torch.float32)
    else:
        emb = torch.tensor(emb, dtype=torch.float32)
    if edge_index is None or len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(np.array(edge_index, dtype=np.int64).T, dtype=torch.long)

    data = Data()
    data.x = x
    data.edge_index = edge_index
    data.emb = emb
    data.y = torch.tensor([float(y)], dtype=torch.float32)
    data.rna_len = torch.tensor([len(sequence)], dtype=torch.long)
    data.e_id = torch.tensor([int(entry_id)], dtype=torch.long)
    data.t_id = str(target_id)
    return data


class PDBContactPairDataset(Dataset):
    """Dataset of prebuilt RNA-ligand contact pair .pt files."""

    def __init__(self, root):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt contact samples found under {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = load_torch(self.files[index])
        return item["rna"], item["molecule"], item["contact_map"].float(), item.get("meta", {})


def contact_collate(items):
    rnas, molecules, contacts, metas = zip(*items)
    rna_batch = Batch.from_data_list(list(rnas))
    molecule_batch = Batch.from_data_list(list(molecules))

    max_rna_len = min(512, max(contact.size(0) for contact in contacts))
    max_atom_len = min(128, max(contact.size(1) for contact in contacts))
    target = torch.zeros((len(contacts), max_rna_len, max_atom_len), dtype=torch.float32)
    mask = torch.zeros_like(target)

    for i, contact in enumerate(contacts):
        r = min(max_rna_len, contact.size(0))
        a = min(max_atom_len, contact.size(1))
        target[i, :r, :a] = contact[:r, :a]
        mask[i, :r, :a] = 1.0

    return rna_batch, molecule_batch, target, mask, list(metas)
