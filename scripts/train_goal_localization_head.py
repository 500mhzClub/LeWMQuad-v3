#!/usr/bin/env python3
"""Train a regression head to predict goal vectors from frozen features."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class GoalLocalizationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, start_feat, goal_feat):
        x = torch.cat([start_feat, goal_feat], dim=-1)
        return self.net(x)

def load_data(path: Path):
    data = np.load(path)
    mask = data["goal_present"]
    start_feat = torch.from_numpy(data["start_raw"][mask]).float()
    goal_feat = torch.from_numpy(data["goal_raw"][mask]).float()
    targets = torch.from_numpy(data["relative_goal_vector_body"][mask]).float()
    return start_feat, goal_feat, targets

def evaluate(head, start, goal, targets, device):
    head.eval()
    with torch.no_grad():
        preds = head(start.to(device), goal.to(device)).cpu()
    
    mse = float(torch.nn.functional.mse_loss(preds, targets))
    
    # distance error
    pred_dist = torch.norm(preds, dim=1)
    targ_dist = torch.norm(targets, dim=1)
    dist_error = float(torch.abs(pred_dist - targ_dist).mean())
    
    # angular error
    pred_angle = torch.atan2(preds[:, 1], preds[:, 0])
    targ_angle = torch.atan2(targets[:, 1], targets[:, 0])
    
    # normalize angle difference to [-pi, pi]
    angle_diff = torch.remainder(pred_angle - targ_angle + np.pi, 2 * np.pi) - np.pi
    angle_error = float(torch.abs(angle_diff).mean()) * (180.0 / np.pi) # in degrees
    
    # R2
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_start, train_goal, train_targets = load_data(args.train_data)
    val_start, val_goal, val_targets = load_data(args.val_data)
    
    device = torch.device(args.device)
    head = GoalLocalizationHead(input_dim=train_start.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    
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
