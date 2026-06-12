#!/usr/bin/env python3
"""Train a cross-attention head to predict goal vectors from spatial patches."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class PatchCrossAttentionHead(nn.Module):
    def __init__(self, token_dim: int = 192, num_patches: int = 64, num_heads: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.token_dim = token_dim
        self.num_patches = num_patches
        
        # Cross attention: Queries come from start image, Keys/Values from goal image
        self.cross_attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        
        # Self attention for refining the matched features
        self.self_attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        
        self.mlp = nn.Sequential(
            nn.Linear(token_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, start_spatial, goal_spatial):
        # Extract CLS and Patches
        # Input shape: [batch, cls_dim + num_patches * token_dim]
        start_cls = start_spatial[:, :self.token_dim]
        goal_cls = goal_spatial[:, :self.token_dim]
        
        start_patches = start_spatial[:, self.token_dim:].view(-1, self.num_patches, self.token_dim)
        goal_patches = goal_spatial[:, self.token_dim:].view(-1, self.num_patches, self.token_dim)
        
        # 1. Cross Attention: "Where in the goal image are the features from my current view?"
        # Q: start_patches, K, V: goal_patches
        attn_out, _ = self.cross_attn(query=start_patches, key=goal_patches, value=goal_patches)
        
        # Residual + Norm
        x = start_patches + attn_out
        
        # 2. Self Attention to integrate global layout
        self_out, _ = self.self_attn(query=x, key=x, value=x)
        x = x + self_out
        
        # 3. Pool the spatial representation
        pooled = x.mean(dim=1)
        
        # 4. Concatenate with start CLS to ground the regression
        features = torch.cat([start_cls, pooled], dim=-1)
        
        return self.mlp(features)

def load_data(path: Path):
    data = np.load(path)
    mask = data["goal_present"]
    start_feat = torch.from_numpy(data["start_spatial"][mask]).float()
    goal_feat = torch.from_numpy(data["goal_spatial"][mask]).float()
    targets = torch.from_numpy(data["relative_goal_vector_body"][mask]).float()
    return start_feat, goal_feat, targets

def evaluate(head, start, goal, targets, device):
    head.eval()
    with torch.no_grad():
        # process in batches to save memory
        preds = []
        for i in range(0, len(start), 512):
            p = head(start[i:i+512].to(device), goal[i:i+512].to(device)).cpu()
            preds.append(p)
        preds = torch.cat(preds, dim=0)
    
    mse = float(torch.nn.functional.mse_loss(preds, targets))
    
    pred_dist = torch.norm(preds, dim=1)
    targ_dist = torch.norm(targets, dim=1)
    dist_error = float(torch.abs(pred_dist - targ_dist).mean())
    
    pred_angle = torch.atan2(preds[:, 1], preds[:, 0])
    targ_angle = torch.atan2(targets[:, 1], targets[:, 0])
    
    angle_diff = torch.remainder(pred_angle - targ_angle + np.pi, 2 * np.pi) - np.pi
    angle_error = float(torch.abs(angle_diff).mean()) * (180.0 / np.pi)
    
    ss_res = float(((preds - targets) ** 2).sum())
    ss_tot = float(((targets - targets.mean(dim=0)) ** 2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        "mse": mse,
        "dist_error_m": dist_error,
        "angular_error_deg": angle_error,
        "r2": r2
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_start, train_goal, train_targets = load_data(args.train_data)
    val_start, val_goal, val_targets = load_data(args.val_data)
    
    device = torch.device(args.device)
    head = PatchCrossAttentionHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    
    dataset = torch.utils.data.TensorDataset(train_start, train_goal, train_targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    for epoch in range(args.epochs):
        head.train()
        total_loss = 0
        for s, g, t in loader:
            optimizer.zero_grad()
            preds = head(s.to(device), g.to(device))
            loss = torch.nn.functional.mse_loss(preds, t.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            val_metrics = evaluate(head, val_start, val_goal, val_targets, device)
            logging.info(f"Epoch {epoch+1:03d} | Train MSE: {total_loss/len(loader):.4f} | Val MSE: {val_metrics['mse']:.4f} | Angle Err: {val_metrics['angular_error_deg']:.1f}° | R2: {val_metrics['r2']:.3f}")
            
    val_metrics = evaluate(head, val_start, val_goal, val_targets, device)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), args.output)
    
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(val_metrics, indent=2))
    print(json.dumps(val_metrics, indent=2))
    
if __name__ == "__main__":
    main()
