import torch
import torch.nn as nn

from .gnn_model_rna import RNA_feature_extraction
from .gnn_model_mole import GCNNet as GNN_molecule
from .transformer_encoder import transformer_1d as mole_seq_model
from .cross_attention import cross_attention


class ContactHead(nn.Module):
    """Predict nucleotide-atom contacts from cross-fused token embeddings."""

    def __init__(self, hidden_dim, pair_hidden_dim=128, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, pair_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_dim, pair_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_dim // 2, 1),
        )

    def forward(self, rna_tokens, atom_tokens):
        # rna_tokens: B x R x H, atom_tokens: B x A x H
        rna_pair = rna_tokens.unsqueeze(2).expand(-1, -1, atom_tokens.size(1), -1)
        atom_pair = atom_tokens.unsqueeze(1).expand(-1, rna_tokens.size(1), -1, -1)
        pair = torch.cat((rna_pair, atom_pair, rna_pair * atom_pair), dim=-1)
        return self.mlp(pair).squeeze(-1)


class PairEnergyHead(nn.Module):
    """Estimate local nucleotide-atom interaction energy for each pair."""

    def __init__(self, hidden_dim, pair_hidden_dim=128, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, pair_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_dim, pair_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_dim // 2, 1),
        )

    def forward(self, rna_tokens, atom_tokens):
        # rna_tokens: B x R x H, atom_tokens: B x A x H
        rna_pair = rna_tokens.unsqueeze(2).expand(-1, -1, atom_tokens.size(1), -1)
        atom_pair = atom_tokens.unsqueeze(1).expand(-1, rna_tokens.size(1), -1, -1)
        pair = torch.cat(
            (rna_pair, atom_pair, rna_pair * atom_pair, torch.abs(rna_pair - atom_pair)),
            dim=-1,
        )
        return self.mlp(pair).squeeze(-1)


class ContactAwareMultiviewFusion(nn.Module):
    """Refine four global views with context-aware fused-value attention.

    The four inputs are RNA sequence, RNA graph, molecule sequence, and
    molecule graph vectors. A compact contact-prior summary is used to gate the
    shared value vector, so the interaction module is aware of likely physical
    RNA-ligand contact strength.
    """

    def __init__(self, hidden_dim, num_layers=2, dropout=0.1, contact_stat_dim=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5
        self.q_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.k_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fused_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.value_gate = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 4 + contact_stat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 4),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_attn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, views, contact_stats):
        # views: B x 4 x H, contact_stats: B x 4
        x = views
        last_gate = None
        for layer_idx in range(len(self.q_proj)):
            gate_input = torch.cat((x.flatten(start_dim=1), contact_stats), dim=1)
            value_gate = torch.softmax(self.value_gate[layer_idx](gate_input), dim=1)
            fused_value = (x * value_gate.unsqueeze(-1)).sum(dim=1)
            fused_value = self.fused_proj[layer_idx](fused_value).unsqueeze(1)

            query = self.q_proj[layer_idx](x)
            key = self.k_proj[layer_idx](x)
            value = self.v_proj[layer_idx](x) + fused_value
            attention = torch.softmax(torch.matmul(query, key.transpose(1, 2)) * self.scale, dim=-1)
            update = torch.matmul(attention, value)
            x = self.norm_attn[layer_idx](x + self.dropout(update))
            x = self.norm_ffn[layer_idx](x + self.dropout(self.ffn[layer_idx](x)))
            last_gate = value_gate
        return x, last_gate


class DeepRSMAContact(nn.Module):
    """DeepRSMA backbone with optional nucleotide-atom contact guidance."""

    def __init__(
        self,
        hidden_dim=128,
        dropout=0.2,
        contact_guided=False,
        contact_mode=None,
        contact_chunk_size=64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        if contact_mode is None:
            contact_mode = "naive" if contact_guided else "none"
        if contact_mode not in {
            "none",
            "naive",
            "residual",
            "pair_energy",
            "cmif",
            "cmif_residual",
        }:
            raise ValueError(f"Unsupported contact_mode: {contact_mode}")
        self.contact_mode = contact_mode
        self.contact_guided = contact_mode != "none"
        self.contact_chunk_size = contact_chunk_size

        self.rna_graph_model = RNA_feature_extraction(hidden_dim)
        self.mole_graph_model = GNN_molecule(hidden_dim)
        self.mole_seq_model = mole_seq_model(hidden_dim)
        self.cross_attention = cross_attention(hidden_dim)

        self.line1 = nn.Linear(hidden_dim * 2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.guided_line1 = nn.Linear(hidden_dim * 4, 1024)
        self.guided_line2 = nn.Linear(1024, 512)
        self.guided_line3 = nn.Linear(512, 1)
        self.residual_line1 = nn.Linear(hidden_dim * 4, 512)
        self.residual_line2 = nn.Linear(512, 128)
        self.residual_line3 = nn.Linear(128, 1)
        self.pair_energy_head = PairEnergyHead(hidden_dim)
        self.pair_energy_delta = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.cmif_fusion = ContactAwareMultiviewFusion(hidden_dim, num_layers=2, dropout=dropout)
        self.cmif_line1 = nn.Linear(hidden_dim * 4 + 4, 1024)
        self.cmif_line2 = nn.Linear(1024, 512)
        self.cmif_line3 = nn.Linear(512, 1)
        self.cmif_delta_line1 = nn.Linear(hidden_dim * 4 + 4, 512)
        self.cmif_delta_line2 = nn.Linear(512, 128)
        self.cmif_delta_line3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)

        self.rna1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.mole1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.rna2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.mole2 = nn.Linear(hidden_dim * 4, hidden_dim)

        self.relu = nn.ReLU()
        self.contact_head = ContactHead(hidden_dim)
        self._init_residual_head()
        self._init_pair_energy_head()
        self._init_cmif_residual_head()

    def _init_residual_head(self):
        nn.init.zeros_(self.residual_line3.weight)
        nn.init.zeros_(self.residual_line3.bias)

    def _init_pair_energy_head(self):
        last_layer = self.pair_energy_delta[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def _init_cmif_residual_head(self):
        nn.init.zeros_(self.cmif_delta_line3.weight)
        nn.init.zeros_(self.cmif_delta_line3.bias)

    def _pad_molecule_graph_tokens(self, mole_graph_emb, graph_len, device):
        atom_tokens = []
        mask = []
        offset = 0
        for graph_size in graph_len:
            count = int(graph_size.item() if torch.is_tensor(graph_size) else graph_size)
            count = min(count, 128)
            x = mole_graph_emb[offset:offset + count]
            offset += int(graph_size.item() if torch.is_tensor(graph_size) else graph_size)
            if x.size(0) < 128:
                pad = torch.zeros((128 - x.size(0), self.hidden_dim), device=device)
                x = torch.cat((x, pad), dim=0)
            atom_tokens.append(x)
            mask.append([1] * count + [0] * (128 - count))
        return torch.stack(atom_tokens).to(device), torch.tensor(mask, dtype=torch.float, device=device)

    def encode(self, rna_batch, mole_batch, device):
        rna_out_seq, rna_out_graph, rna_mask_seq, rna_mask_graph, rna_seq_final, rna_graph_final = (
            self.rna_graph_model(rna_batch, device)
        )

        mole_graph_emb, mole_graph_final = self.mole_graph_model(mole_batch)
        mole_seq_emb, _, mole_mask_seq = self.mole_seq_model(mole_batch, device)
        mole_seq_final = (
            mole_seq_emb[-1] * mole_mask_seq.to(device).unsqueeze(dim=2)
        ).mean(dim=1).squeeze(dim=1)

        mole_out_graph, mole_mask_graph = self._pad_molecule_graph_tokens(
            mole_graph_emb, mole_batch.graph_len, device
        )

        context_layer, attention_score = self.cross_attention(
            [rna_out_seq, rna_out_graph, mole_seq_emb[-1], mole_out_graph],
            [
                rna_mask_seq.to(device),
                rna_mask_graph.to(device),
                mole_mask_seq.to(device),
                mole_mask_graph.to(device),
            ],
            device,
        )

        out_rna = context_layer[-1][0]
        out_mole = context_layer[-1][1]

        return {
            "out_rna": out_rna,
            "out_mole": out_mole,
            "rna_mask_seq": rna_mask_seq.to(device),
            "rna_mask_graph": rna_mask_graph.to(device),
            "mole_mask_seq": mole_mask_seq.to(device),
            "mole_mask_graph": mole_mask_graph.to(device),
            "rna_seq_final": rna_seq_final,
            "rna_graph_final": rna_graph_final,
            "mole_seq_final": mole_seq_final,
            "mole_graph_final": mole_graph_final,
            "attention_score": attention_score,
        }

    def cross_features_from_encoded(self, encoded):
        out_rna = encoded["out_rna"]
        out_mole = encoded["out_mole"]
        rna_mask_seq = encoded["rna_mask_seq"]
        rna_mask_graph = encoded["rna_mask_graph"]
        mole_mask_seq = encoded["mole_mask_seq"]
        mole_mask_graph = encoded["mole_mask_graph"]

        rna_cross_seq = (
            (out_rna[:, 0:512] * rna_mask_seq.unsqueeze(dim=2)).mean(dim=1).squeeze(dim=1)
            + encoded["rna_seq_final"]
        ) / 2
        rna_cross_stru = (
            (out_rna[:, 512:] * rna_mask_graph.unsqueeze(dim=2)).mean(dim=1).squeeze(dim=1)
            + encoded["rna_graph_final"]
        ) / 2
        rna_cross = (rna_cross_seq + rna_cross_stru) / 2
        rna_cross = self.rna2(self.dropout(self.relu(self.rna1(rna_cross))))

        mole_cross_seq = (
            (out_mole[:, 0:128] * mole_mask_seq.unsqueeze(dim=2)).mean(dim=1).squeeze(dim=1)
            + encoded["mole_seq_final"]
        ) / 2
        mole_cross_stru = (
            (out_mole[:, 128:] * mole_mask_graph.unsqueeze(dim=2)).mean(dim=1).squeeze(dim=1)
            + encoded["mole_graph_final"]
        ) / 2
        mole_cross = (mole_cross_seq + mole_cross_stru) / 2
        mole_cross = self.mole2(self.dropout(self.relu(self.mole1(mole_cross))))
        return rna_cross, mole_cross

    def multiview_features_from_encoded(self, encoded):
        out_rna = encoded["out_rna"]
        out_mole = encoded["out_mole"]
        rna_mask_seq = encoded["rna_mask_seq"]
        rna_mask_graph = encoded["rna_mask_graph"]
        mole_mask_seq = encoded["mole_mask_seq"]
        mole_mask_graph = encoded["mole_mask_graph"]

        rna_seq = (
            (out_rna[:, 0:512] * rna_mask_seq.unsqueeze(dim=2)).mean(dim=1).squeeze(dim=1)
            + encoded["rna_seq_final"]
        ) / 2
        rna_graph = (
            (out_rna[:, 512:] * rna_mask_graph.unsqueeze(dim=2)).mean(dim=1).squeeze(dim=1)
            + encoded["rna_graph_final"]
        ) / 2
        mole_seq = (
            (out_mole[:, 0:128] * mole_mask_seq.unsqueeze(dim=2)).mean(dim=1).squeeze(dim=1)
            + encoded["mole_seq_final"]
        ) / 2
        mole_graph = (
            (out_mole[:, 128:] * mole_mask_graph.unsqueeze(dim=2)).mean(dim=1).squeeze(dim=1)
            + encoded["mole_graph_final"]
        ) / 2

        rna_seq = self.rna2(self.dropout(self.relu(self.rna1(rna_seq))))
        rna_graph = self.rna2(self.dropout(self.relu(self.rna1(rna_graph))))
        mole_seq = self.mole2(self.dropout(self.relu(self.mole1(mole_seq))))
        mole_graph = self.mole2(self.dropout(self.relu(self.mole1(mole_graph))))
        return torch.stack((rna_seq, rna_graph, mole_seq, mole_graph), dim=1)

    def affinity_from_encoded(self, encoded):
        rna_cross, mole_cross = self.cross_features_from_encoded(encoded)
        return self.affinity_from_cross_features(rna_cross, mole_cross)

    def affinity_from_cross_features(self, rna_cross, mole_cross):
        out = torch.cat((rna_cross, mole_cross), dim=1)
        out = self.line1(out)
        out = self.dropout(self.relu(out))
        out = self.line2(out)
        out = self.dropout(self.relu(out))
        return self.line3(out)

    def contact_tokens_from_encoded(self, encoded):
        # Fuse RNA sequence/structure token streams at nucleotide positions.
        rna_tokens = (encoded["out_rna"][:, 0:512] + encoded["out_rna"][:, 512:]) / 2
        # Atom-level tokens come from the molecule graph stream.
        atom_tokens = encoded["out_mole"][:, 128:]
        return rna_tokens, atom_tokens

    def contact_from_encoded(self, encoded):
        rna_tokens, atom_tokens = self.contact_tokens_from_encoded(encoded)
        return self.contact_head(rna_tokens, atom_tokens)

    def contact_prior_stats_from_encoded(self, encoded):
        rna_tokens, atom_tokens = self.contact_tokens_from_encoded(encoded)
        rna_mask = encoded["rna_mask_graph"]
        atom_mask = encoded["mole_mask_graph"]

        batch_size = rna_tokens.size(0)
        prob_sum = torch.zeros((batch_size, 1), device=rna_tokens.device)
        valid_pair_sum = torch.zeros((batch_size, 1), device=rna_tokens.device)
        rna_weight = torch.zeros_like(rna_mask)
        atom_weight = torch.zeros_like(atom_mask)
        max_prob = torch.zeros((batch_size, 1), device=rna_tokens.device)

        with torch.no_grad():
            for start in range(0, rna_tokens.size(1), self.contact_chunk_size):
                end = min(start + self.contact_chunk_size, rna_tokens.size(1))
                logits = self.contact_head(rna_tokens[:, start:end], atom_tokens)
                pair_mask = rna_mask[:, start:end].unsqueeze(2) * atom_mask.unsqueeze(1)
                prob = torch.sigmoid(logits) * pair_mask
                prob_sum = prob_sum + prob.sum(dim=(1, 2), keepdim=False).unsqueeze(1)
                valid_pair_sum = valid_pair_sum + pair_mask.sum(dim=(1, 2), keepdim=False).unsqueeze(1)
                rna_weight[:, start:end] = prob.sum(dim=2)
                atom_weight = atom_weight + prob.sum(dim=1)
                masked_prob = prob.masked_fill(pair_mask <= 0, 0.0)
                max_prob = torch.maximum(max_prob, masked_prob.amax(dim=(1, 2), keepdim=False).unsqueeze(1))

        contact_density = prob_sum / valid_pair_sum.clamp_min(1.0)
        rna_focus = rna_weight.amax(dim=1, keepdim=True) / atom_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        atom_focus = atom_weight.amax(dim=1, keepdim=True) / rna_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.cat((contact_density, max_prob, rna_focus, atom_focus), dim=1)

    def contact_guided_features_from_encoded(self, encoded, cross_features=None):
        if cross_features is None:
            rna_cross, mole_cross = self.cross_features_from_encoded(encoded)
        else:
            rna_cross, mole_cross = cross_features
        rna_tokens, atom_tokens = self.contact_tokens_from_encoded(encoded)
        rna_mask = encoded["rna_mask_graph"]
        atom_mask = encoded["mole_mask_graph"]

        # Use the pretrained contact head as a detached structural prior. This
        # keeps pKd fine-tuning memory small while gradients still update the
        # RNA/atom tokens through the contact-weighted pooling below.
        rna_weight_parts = []
        atom_weight = torch.zeros_like(atom_mask)
        with torch.no_grad():
            for start in range(0, rna_tokens.size(1), self.contact_chunk_size):
                end = min(start + self.contact_chunk_size, rna_tokens.size(1))
                logits = self.contact_head(rna_tokens[:, start:end], atom_tokens)
                prob = torch.sigmoid(logits)
                pair_mask = rna_mask[:, start:end].unsqueeze(2) * atom_mask.unsqueeze(1)
                prob = prob * pair_mask
                rna_weight_parts.append(prob.sum(dim=2))
                atom_weight = atom_weight + prob.sum(dim=1)
        rna_weight = torch.cat(rna_weight_parts, dim=1)

        rna_guided = (rna_tokens * rna_weight.unsqueeze(2)).sum(dim=1)
        rna_guided = rna_guided / rna_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        atom_guided = (atom_tokens * atom_weight.unsqueeze(2)).sum(dim=1)
        atom_guided = atom_guided / atom_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)

        return torch.cat((rna_cross, mole_cross, rna_guided, atom_guided), dim=1)

    def contact_guided_affinity_from_encoded(self, encoded):
        out = self.contact_guided_features_from_encoded(encoded)
        out = self.guided_line1(out)
        out = self.dropout(self.relu(out))
        out = self.guided_line2(out)
        out = self.dropout(self.relu(out))
        return self.guided_line3(out)

    def contact_residual_affinity_from_encoded(self, encoded):
        rna_cross, mole_cross = self.cross_features_from_encoded(encoded)
        base_affinity = self.affinity_from_cross_features(rna_cross, mole_cross)
        guided_features = self.contact_guided_features_from_encoded(
            encoded, cross_features=(rna_cross, mole_cross)
        )
        delta = self.residual_line1(guided_features)
        delta = self.dropout(self.relu(delta))
        delta = self.residual_line2(delta)
        delta = self.dropout(self.relu(delta))
        return base_affinity + self.residual_line3(delta)

    def pair_energy_affinity_from_encoded(self, encoded):
        rna_cross, mole_cross = self.cross_features_from_encoded(encoded)
        base_affinity = self.affinity_from_cross_features(rna_cross, mole_cross)
        rna_tokens, atom_tokens = self.contact_tokens_from_encoded(encoded)
        rna_mask = encoded["rna_mask_graph"]
        atom_mask = encoded["mole_mask_graph"]

        batch_size = rna_tokens.size(0)
        energy_sum = torch.zeros((batch_size, 1), device=rna_tokens.device)
        gate_sum = torch.zeros((batch_size, 1), device=rna_tokens.device)
        valid_pair_sum = torch.zeros((batch_size, 1), device=rna_tokens.device)
        chunk_size = min(self.contact_chunk_size, 32)

        for start in range(0, rna_tokens.size(1), chunk_size):
            end = min(start + chunk_size, rna_tokens.size(1))
            rna_chunk = rna_tokens[:, start:end]
            pair_mask = rna_mask[:, start:end].unsqueeze(2) * atom_mask.unsqueeze(1)

            # Treat the pretrained contact head as a structural gate. The local
            # energy path remains trainable while the gate is kept stable.
            with torch.no_grad():
                contact_logits = self.contact_head(rna_chunk, atom_tokens)
                gate = torch.sigmoid(contact_logits) * pair_mask

            local_energy = self.pair_energy_head(rna_chunk, atom_tokens)
            energy_sum = energy_sum + (local_energy * gate).sum(dim=(1, 2), keepdim=False).unsqueeze(1)
            gate_sum = gate_sum + gate.sum(dim=(1, 2), keepdim=False).unsqueeze(1)
            valid_pair_sum = valid_pair_sum + pair_mask.sum(dim=(1, 2), keepdim=False).unsqueeze(1)

        contact_energy = energy_sum / gate_sum.clamp_min(1e-6)
        contact_mass = gate_sum / valid_pair_sum.clamp_min(1e-6)
        delta_input = torch.cat((rna_cross, mole_cross, contact_energy, contact_mass), dim=1)
        return base_affinity + self.pair_energy_delta(delta_input)

    def cmif_affinity_from_encoded(self, encoded):
        views = self.multiview_features_from_encoded(encoded)
        contact_stats = self.contact_prior_stats_from_encoded(encoded)
        refined_views, _ = self.cmif_fusion(views, contact_stats)
        out = torch.cat((refined_views.flatten(start_dim=1), contact_stats), dim=1)
        out = self.cmif_line1(out)
        out = self.dropout(self.relu(out))
        out = self.cmif_line2(out)
        out = self.dropout(self.relu(out))
        return self.cmif_line3(out)

    def cmif_residual_affinity_from_encoded(self, encoded):
        rna_cross, mole_cross = self.cross_features_from_encoded(encoded)
        base_affinity = self.affinity_from_cross_features(rna_cross, mole_cross)

        views = self.multiview_features_from_encoded(encoded)
        contact_stats = self.contact_prior_stats_from_encoded(encoded)
        refined_views, _ = self.cmif_fusion(views, contact_stats)
        delta = torch.cat((refined_views.flatten(start_dim=1), contact_stats), dim=1)
        delta = self.cmif_delta_line1(delta)
        delta = self.dropout(self.relu(delta))
        delta = self.cmif_delta_line2(delta)
        delta = self.dropout(self.relu(delta))
        return base_affinity + self.cmif_delta_line3(delta)

    def forward(self, rna_batch, mole_batch, device=None, return_contact=False):
        if device is None:
            device = next(self.parameters()).device
        encoded = self.encode(rna_batch, mole_batch, device)
        if self.contact_mode == "naive":
            affinity = self.contact_guided_affinity_from_encoded(encoded)
        elif self.contact_mode == "residual":
            affinity = self.contact_residual_affinity_from_encoded(encoded)
        elif self.contact_mode == "pair_energy":
            affinity = self.pair_energy_affinity_from_encoded(encoded)
        elif self.contact_mode == "cmif":
            affinity = self.cmif_affinity_from_encoded(encoded)
        elif self.contact_mode == "cmif_residual":
            affinity = self.cmif_residual_affinity_from_encoded(encoded)
        else:
            affinity = self.affinity_from_encoded(encoded)
        if not return_contact:
            return {"affinity": affinity, "encoded": encoded}

        contact_logits = self.contact_from_encoded(encoded)
        return {
            "affinity": affinity,
            "contact_logits": contact_logits,
            "rna_contact_mask": encoded["rna_mask_graph"],
            "atom_contact_mask": encoded["mole_mask_graph"],
            "encoded": encoded,
        }
