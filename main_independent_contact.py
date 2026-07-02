import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from data import (
    Molecule_dataset,
    Molecule_dataset_independent,
    PDBContactPairDataset,
    RNA_dataset,
    RNA_dataset_independent,
    contact_collate,
)
from model import DeepRSMAContact


os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCH = int(os.environ.get("DEEPRSMA_EPOCH", 200))
BATCH_SIZE = int(os.environ.get("DEEPRSMA_BATCH_SIZE", 8))
NUM_WORKERS = int(os.environ.get("DEEPRSMA_NUM_WORKERS", 0 if os.name == "nt" else 1))
MAX_TRAIN = int(os.environ.get("DEEPRSMA_MAX_TRAIN", 0))
MAX_TEST = int(os.environ.get("DEEPRSMA_MAX_TEST", 0))
SEED = int(os.environ.get("DEEPRSMA_SEED", 1))
CONTACT_CKPT = os.environ.get("DEEPRSMA_CONTACT_CKPT", "")
LR = float(os.environ.get("DEEPRSMA_LR", 6e-5))
WEIGHT_DECAY = float(os.environ.get("DEEPRSMA_WEIGHT_DECAY", 1e-5))
SAVE_TAG = os.environ.get("DEEPRSMA_SAVE_TAG", "").strip()
FREEZE_BACKBONE_EPOCHS = int(os.environ.get("DEEPRSMA_FREEZE_BACKBONE_EPOCHS", 0))
SKIP_AFFINITY_HEAD = int(os.environ.get("DEEPRSMA_SKIP_AFFINITY_HEAD", 1))
CONTACT_GUIDED_ENV = os.environ.get("DEEPRSMA_CONTACT_GUIDED", "0").strip().lower()
CONTACT_MODE = os.environ.get("DEEPRSMA_CONTACT_MODE", "").strip().lower()
if not CONTACT_MODE:
    CONTACT_MODE = "naive" if CONTACT_GUIDED_ENV in {"1", "true", "yes", "naive"} else "none"
if CONTACT_MODE == "guided":
    CONTACT_MODE = "naive"
CONTACT_REG_DIR = os.environ.get("DEEPRSMA_CONTACT_REG_DIR", "").strip()
CONTACT_REG_WEIGHT = float(os.environ.get("DEEPRSMA_CONTACT_REG_WEIGHT", 0.0))
CONTACT_REG_STEPS = int(os.environ.get("DEEPRSMA_CONTACT_REG_STEPS", 0))
CONTACT_REG_BATCH_SIZE = int(os.environ.get("DEEPRSMA_CONTACT_REG_BATCH_SIZE", 2))
CONTACT_REG_ALPHA = float(os.environ.get("DEEPRSMA_CONTACT_REG_ALPHA", 0.75))
CONTACT_REG_GAMMA = float(os.environ.get("DEEPRSMA_CONTACT_REG_GAMMA", 2.0))
AFFINITY_INIT_CKPT = os.environ.get("DEEPRSMA_AFFINITY_INIT_CKPT", "").strip()
TRAIN_ONLY_CMIF = int(os.environ.get("DEEPRSMA_TRAIN_ONLY_CMIF", 0))
EVAL_BEFORE_TRAIN = int(os.environ.get("DEEPRSMA_EVAL_BEFORE_TRAIN", 0))
os.makedirs("save", exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)


class CustomDualDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        assert len(self.dataset1) == len(self.dataset2)

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]

    def __len__(self):
        return len(self.dataset1)


def load_contact_checkpoint(model, path):
    if not path:
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    skipped = []
    if SKIP_AFFINITY_HEAD:
        affinity_prefixes = (
            "line1.",
            "line2.",
            "line3.",
            "guided_line1.",
            "guided_line2.",
            "guided_line3.",
            "residual_line1.",
            "residual_line2.",
            "residual_line3.",
            "pair_energy_head.",
            "pair_energy_delta.",
            "cmif_fusion.",
            "cmif_line1.",
            "cmif_line2.",
            "cmif_line3.",
            "cmif_delta_line1.",
            "cmif_delta_line2.",
            "cmif_delta_line3.",
            "rna1.",
            "rna2.",
            "mole1.",
            "mole2.",
        )
        filtered_state = {}
        for key, value in state.items():
            if key.startswith(affinity_prefixes):
                skipped.append(key)
            else:
                filtered_state[key] = value
        state = filtered_state
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded contact checkpoint: {path}")
    print(f"Missing keys: {len(missing)} Unexpected keys: {len(unexpected)}")
    if skipped:
        print(f"Skipped untrained affinity-head keys: {len(skipped)}")


def load_affinity_checkpoint(model, path):
    if not path:
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded affinity initialization checkpoint: {path}")
    print(f"Missing keys: {len(missing)} Unexpected keys: {len(unexpected)}")


def set_backbone_trainable(model, trainable):
    backbone_modules = [
        model.rna_graph_model,
        model.mole_graph_model,
        model.mole_seq_model,
        model.cross_attention,
        model.contact_head,
    ]
    for module in backbone_modules:
        for param in module.parameters():
            param.requires_grad = trainable


def set_only_cmif_trainable(model):
    for param in model.parameters():
        param.requires_grad = False
    trainable_prefixes = (
        "cmif_fusion.",
        "cmif_delta_line1.",
        "cmif_delta_line2.",
        "cmif_delta_line3.",
    )
    for name, param in model.named_parameters():
        if name.startswith(trainable_prefixes):
            param.requires_grad = True


def trainable_parameter_count(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def masked_focal_loss(logits, target, mask, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)
    alpha_t = torch.where(target > 0.5, alpha, 1.0 - alpha)
    loss = alpha_t * torch.pow(1.0 - pt, gamma) * bce
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def next_contact_batch(contact_iter, contact_loader):
    try:
        batch = next(contact_iter)
    except StopIteration:
        contact_iter = iter(contact_loader)
        batch = next(contact_iter)
    return batch, contact_iter


def run_contact_regularization(model, optimizer, contact_loader, contact_iter, steps):
    if contact_loader is None or steps <= 0:
        return 0.0, contact_iter

    model.train()
    total_loss = 0.0
    for _ in range(steps):
        (rna_batch, molecule_batch, target, mask, _), contact_iter = next_contact_batch(
            contact_iter, contact_loader
        )
        rna_batch = rna_batch.to(device)
        molecule_batch = molecule_batch.to(device)
        target = target.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        output = model(rna_batch, molecule_batch, device=device, return_contact=True)
        logits = output["contact_logits"][:, : target.size(1), : target.size(2)]
        loss = masked_focal_loss(
            logits,
            target,
            mask,
            alpha=CONTACT_REG_ALPHA,
            gamma=CONTACT_REG_GAMMA,
        )
        (CONTACT_REG_WEIGHT * loss).backward()
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(steps, 1), contact_iter


def evaluate(model, loader, loss_fct):
    model.eval()
    y_label = []
    y_pred = []
    test_loss = 0.0
    with torch.no_grad():
        for batch_rna, batch_mole in loader:
            label = batch_rna.y.detach().cpu().float()
            score = model(batch_rna.to(device), batch_mole.to(device), device=device)["affinity"]
            pred = torch.squeeze(score, 1)
            loss_t = loss_fct(pred.cpu(), label)
            y_label.extend(label.numpy().flatten().tolist())
            y_pred.extend(torch.squeeze(score).detach().cpu().numpy().flatten().tolist())
            test_loss += float(loss_t.item())
    p = pearsonr(y_label, y_pred)
    s = spearmanr(y_label, y_pred)
    rmse = np.sqrt(mean_squared_error(y_label, y_pred))
    model.train()
    return p[0], s[0], rmse, test_loss


def main():
    set_seed(SEED)

    rna_dataset = RNA_dataset("Viral_RNA_independent")
    molecule_dataset = Molecule_dataset("Viral_RNA_independent")
    rna_dataset_in = RNA_dataset_independent()
    molecule_dataset_in = Molecule_dataset_independent()

    if MAX_TRAIN:
        rna_dataset = rna_dataset[:MAX_TRAIN]
        molecule_dataset = molecule_dataset[:MAX_TRAIN]
    if MAX_TEST:
        rna_dataset_in = rna_dataset_in[:MAX_TEST]
        molecule_dataset_in = molecule_dataset_in[:MAX_TEST]

    train_dataset = CustomDualDataset(rna_dataset, molecule_dataset)
    test_dataset = CustomDualDataset(rna_dataset_in, molecule_dataset_in)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=False,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=NUM_WORKERS,
        drop_last=False,
        shuffle=False,
    )

    contact_loader = None
    contact_iter = None
    if CONTACT_REG_DIR and CONTACT_REG_WEIGHT > 0 and CONTACT_REG_STEPS > 0:
        contact_dataset = PDBContactPairDataset(CONTACT_REG_DIR)
        contact_loader = TorchDataLoader(
            contact_dataset,
            batch_size=CONTACT_REG_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=contact_collate,
        )
        contact_iter = iter(contact_loader)

    model = DeepRSMAContact(contact_mode=CONTACT_MODE).to(device)
    load_contact_checkpoint(model, CONTACT_CKPT)
    load_affinity_checkpoint(model, AFFINITY_INIT_CKPT)

    print(
        f"Fine-tune config: seed={SEED} epochs={EPOCH} batch_size={BATCH_SIZE} "
        f"lr={LR} weight_decay={WEIGHT_DECAY} freeze_backbone_epochs={FREEZE_BACKBONE_EPOCHS} "
        f"contact_mode={CONTACT_MODE} contact_reg_dir={CONTACT_REG_DIR or 'none'} "
        f"contact_reg_weight={CONTACT_REG_WEIGHT} contact_reg_steps={CONTACT_REG_STEPS} "
        f"affinity_init_ckpt={AFFINITY_INIT_CKPT or 'none'} train_only_cmif={TRAIN_ONLY_CMIF}"
    )
    if TRAIN_ONLY_CMIF:
        set_only_cmif_trainable(model)
        print(f"Only CMIF residual adapter trainable. Trainable parameters: {trainable_parameter_count(model)}")
    if FREEZE_BACKBONE_EPOCHS > 0:
        set_backbone_trainable(model, False)
        print(f"Backbone frozen. Trainable parameters: {trainable_parameter_count(model)}")

    optimizer = optim.Adam((param for param in model.parameters() if param.requires_grad), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fct = nn.MSELoss()
    max_p = -1.0
    save_suffix = f"_{SAVE_TAG}" if SAVE_TAG else ""
    save_path = f"save/model_independent_contact_{SEED}{save_suffix}.pth"

    if EVAL_BEFORE_TRAIN:
        p, s, rmse, _ = evaluate(model, test_loader, loss_fct)
        print("Initial:", "pcc:", p, "scc:", s, "rmse:", rmse)
        if max_p < p:
            max_p = p
            print(" ")
            print("Best:", "epo:", "init", "pcc:", p, "scc: ", s, "rmse:", rmse)
            torch.save(model.state_dict(), save_path)

    for epoch in range(EPOCH):
        if FREEZE_BACKBONE_EPOCHS > 0 and epoch == FREEZE_BACKBONE_EPOCHS:
            set_backbone_trainable(model, True)
            print(f"Backbone unfrozen at epoch {epoch}. Trainable parameters: {trainable_parameter_count(model)}")

        train_loss = 0.0
        for batch_rna, batch_mole in train_loader:
            optimizer.zero_grad()
            pred = model(batch_rna.to(device), batch_mole.to(device), device=device)["affinity"]
            loss = loss_fct(pred.squeeze(dim=1), batch_rna.y.float())
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        contact_reg_loss = 0.0
        if epoch >= FREEZE_BACKBONE_EPOCHS:
            contact_reg_loss, contact_iter = run_contact_regularization(
                model,
                optimizer,
                contact_loader,
                contact_iter,
                CONTACT_REG_STEPS,
            )
        p, s, rmse, _ = evaluate(model, test_loader, loss_fct)
        print(
            "epo:",
            epoch,
            "pcc:",
            p,
            "scc: ",
            s,
            "rmse:",
            rmse,
            "contact_reg_loss:",
            contact_reg_loss,
        )
        if max_p < p:
            max_p = p
            print(" ")
            print("Best:", "epo:", epoch, "pcc:", p, "scc: ", s, "rmse:", rmse)
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
