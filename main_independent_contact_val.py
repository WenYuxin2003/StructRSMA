import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader

from data import RNA_dataset, Molecule_dataset, RNA_dataset_independent, Molecule_dataset_independent
from model import DeepRSMAContact


os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCH = int(os.environ.get("DEEPRSMA_EPOCH", 200))
BATCH_SIZE = int(os.environ.get("DEEPRSMA_BATCH_SIZE", 8))
NUM_WORKERS = int(os.environ.get("DEEPRSMA_NUM_WORKERS", 0 if os.name == "nt" else 1))
MAX_TRAIN = int(os.environ.get("DEEPRSMA_MAX_TRAIN", 0))
MAX_TEST = int(os.environ.get("DEEPRSMA_MAX_TEST", 0))
SEED = int(os.environ.get("DEEPRSMA_SEED", 1))
VAL_RATIO = float(os.environ.get("DEEPRSMA_VAL_RATIO", 0.1))
VAL_SIZE = int(os.environ.get("DEEPRSMA_VAL_SIZE", 0))
VAL_SELECTION = os.environ.get("DEEPRSMA_VAL_SELECTION", "pcc").strip().lower()
REFIT_FULL = int(os.environ.get("DEEPRSMA_REFIT_FULL", 0))
CONTACT_CKPT = os.environ.get("DEEPRSMA_CONTACT_CKPT", "")
LR = float(os.environ.get("DEEPRSMA_LR", 6e-5))
WEIGHT_DECAY = float(os.environ.get("DEEPRSMA_WEIGHT_DECAY", 1e-5))
SAVE_TAG = os.environ.get("DEEPRSMA_SAVE_TAG", "").strip()
FREEZE_BACKBONE_EPOCHS = int(os.environ.get("DEEPRSMA_FREEZE_BACKBONE_EPOCHS", 0))
SHUFFLE_TRAIN = int(os.environ.get("DEEPRSMA_SHUFFLE_TRAIN", 0))
SKIP_AFFINITY_HEAD = int(os.environ.get("DEEPRSMA_SKIP_AFFINITY_HEAD", 1))
CONTACT_GUIDED_ENV = os.environ.get("DEEPRSMA_CONTACT_GUIDED", "0").strip().lower()
CONTACT_MODE = os.environ.get("DEEPRSMA_CONTACT_MODE", "").strip().lower()
if not CONTACT_MODE:
    CONTACT_MODE = "naive" if CONTACT_GUIDED_ENV in {"1", "true", "yes", "naive"} else "none"
if CONTACT_MODE == "guided":
    CONTACT_MODE = "naive"
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


def split_train_val(dataset):
    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)
    if VAL_SIZE > 0:
        val_count = VAL_SIZE
    else:
        val_count = int(round(len(indices) * VAL_RATIO))
    val_count = max(1, min(val_count, len(indices) - 1))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


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


def trainable_parameter_count(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def evaluate(model, loader, loss_fct):
    model.eval()
    y_label = []
    y_pred = []
    total_loss = 0.0
    with torch.no_grad():
        for batch_rna, batch_mole in loader:
            label = batch_rna.y.detach().cpu().float()
            score = model(batch_rna.to(device), batch_mole.to(device), device=device)["affinity"]
            pred = torch.squeeze(score, 1)
            loss_t = loss_fct(pred.cpu(), label)
            y_label.extend(label.numpy().flatten().tolist())
            y_pred.extend(torch.squeeze(score).detach().cpu().numpy().flatten().tolist())
            total_loss += float(loss_t.item())
    p = pearsonr(y_label, y_pred)
    s = spearmanr(y_label, y_pred)
    rmse = np.sqrt(mean_squared_error(y_label, y_pred))
    model.train()
    return {
        "pcc": float(p[0]),
        "scc": float(s[0]),
        "rmse": float(rmse),
        "loss": total_loss / max(1, len(loader)),
    }


def is_better(metrics, best_metrics):
    if best_metrics is None:
        return True
    if VAL_SELECTION == "rmse":
        return metrics["rmse"] < best_metrics["rmse"]
    if VAL_SELECTION == "scc":
        return metrics["scc"] > best_metrics["scc"]
    if VAL_SELECTION == "pcc":
        return metrics["pcc"] > best_metrics["pcc"]
    raise ValueError(f"Unsupported DEEPRSMA_VAL_SELECTION: {VAL_SELECTION}")


def clone_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_one_epoch(model, loader, optimizer, loss_fct):
    train_loss = 0.0
    for batch_rna, batch_mole in loader:
        optimizer.zero_grad()
        batch_rna = batch_rna.to(device)
        batch_mole = batch_mole.to(device)
        pred = model(batch_rna, batch_mole, device=device)["affinity"]
        loss = loss_fct(pred.squeeze(dim=1), batch_rna.y.float())
        loss.backward()
        optimizer.step()
        train_loss += float(loss.item())
    return train_loss / max(1, len(loader))


def refit_on_full_train(full_train_dataset, selected_epoch, test_loader, loss_fct, save_path):
    set_seed(SEED)
    refit_loader = DataLoader(
        full_train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=False,
        shuffle=bool(SHUFFLE_TRAIN),
    )
    model = DeepRSMAContact(contact_mode=CONTACT_MODE).to(device)
    load_contact_checkpoint(model, CONTACT_CKPT)
    if FREEZE_BACKBONE_EPOCHS > 0:
        set_backbone_trainable(model, False)
        print(f"Refit backbone frozen. Trainable parameters: {trainable_parameter_count(model)}")
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(selected_epoch + 1):
        if FREEZE_BACKBONE_EPOCHS > 0 and epoch == FREEZE_BACKBONE_EPOCHS:
            set_backbone_trainable(model, True)
            print(
                f"Refit backbone unfrozen at epoch {epoch}. "
                f"Trainable parameters: {trainable_parameter_count(model)}"
            )
        train_loss = train_one_epoch(model, refit_loader, optimizer, loss_fct)
        print("refit_epo:", epoch, "train_loss:", train_loss)

    torch.save(model.state_dict(), save_path)
    return evaluate(model, test_loader, loss_fct)


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

    full_train_dataset = CustomDualDataset(rna_dataset, molecule_dataset)
    train_dataset, val_dataset = split_train_val(full_train_dataset)
    test_dataset = CustomDualDataset(rna_dataset_in, molecule_dataset_in)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=False,
        shuffle=bool(SHUFFLE_TRAIN),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
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

    model = DeepRSMAContact(contact_mode=CONTACT_MODE).to(device)
    load_contact_checkpoint(model, CONTACT_CKPT)

    print(
        f"Val fine-tune config: seed={SEED} epochs={EPOCH} batch_size={BATCH_SIZE} "
        f"lr={LR} weight_decay={WEIGHT_DECAY} val_ratio={VAL_RATIO} val_size={len(val_dataset)} "
        f"val_selection={VAL_SELECTION} freeze_backbone_epochs={FREEZE_BACKBONE_EPOCHS} "
        f"shuffle_train={bool(SHUFFLE_TRAIN)} refit_full={bool(REFIT_FULL)} "
        f"contact_mode={CONTACT_MODE}"
    )
    if FREEZE_BACKBONE_EPOCHS > 0:
        set_backbone_trainable(model, False)
        print(f"Backbone frozen. Trainable parameters: {trainable_parameter_count(model)}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fct = nn.MSELoss()
    best_val = None
    best_epoch = -1
    best_state = None
    save_suffix = f"_{SAVE_TAG}" if SAVE_TAG else ""
    save_path = f"save/model_independent_contact_val_{SEED}{save_suffix}.pth"

    for epoch in range(EPOCH):
        if FREEZE_BACKBONE_EPOCHS > 0 and epoch == FREEZE_BACKBONE_EPOCHS:
            set_backbone_trainable(model, True)
            print(f"Backbone unfrozen at epoch {epoch}. Trainable parameters: {trainable_parameter_count(model)}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fct)

        val_metrics = evaluate(model, val_loader, loss_fct)
        print(
            "epo:",
            epoch,
            "train_loss:",
            train_loss,
            "val_pcc:",
            val_metrics["pcc"],
            "val_scc:",
            val_metrics["scc"],
            "val_rmse:",
            val_metrics["rmse"],
        )
        if is_better(val_metrics, best_val):
            best_val = val_metrics
            best_epoch = epoch
            best_state = clone_state_dict(model)
            torch.save(model.state_dict(), save_path)
            print(
                "BestVal:",
                "epo:",
                epoch,
                "pcc:",
                val_metrics["pcc"],
                "scc:",
                val_metrics["scc"],
                "rmse:",
                val_metrics["rmse"],
            )

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, loss_fct)
    print(
        "FinalTest:",
        "selected_epoch:",
        best_epoch,
        "pcc:",
        test_metrics["pcc"],
        "scc:",
        test_metrics["scc"],
        "rmse:",
        test_metrics["rmse"],
    )
    if REFIT_FULL:
        refit_save_path = f"save/model_independent_contact_val_refit_{SEED}{save_suffix}.pth"
        refit_metrics = refit_on_full_train(
            full_train_dataset,
            best_epoch,
            test_loader,
            loss_fct,
            refit_save_path,
        )
        print(
            "RefitFinalTest:",
            "selected_epoch:",
            best_epoch,
            "pcc:",
            refit_metrics["pcc"],
            "scc:",
            refit_metrics["scc"],
            "rmse:",
            refit_metrics["rmse"],
        )


if __name__ == "__main__":
    main()
