import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from industrial_semantic.config import load_config
from industrial_semantic.dataset import ToolWearDataset
from industrial_semantic.model import SemanticSenderModel
from industrial_semantic.utils import (
    compute_class_weights_from_hist,
    get_logger,
    macro_classification_metrics,
    set_seed,
)


def build_contact_hist(dataset: ToolWearDataset):
    hist = dataset.label_histogram("contact_label", num_classes=3)
    if hist.sum() == 0:
        mapped_hist = dataset.label_histogram("mapped_class", num_classes=4)
        if mapped_hist.sum() > 0:
            hist = np.array([mapped_hist[0], 1, mapped_hist[1:].sum()], dtype=np.int64)
    return hist


def build_wear_hist(dataset: ToolWearDataset):
    hist = dataset.label_histogram("wear_label", num_classes=3, ignore_value=-1)
    if hist.sum() == 0:
        mapped_hist = dataset.label_histogram("mapped_class", num_classes=4)
        if mapped_hist.sum() > 0:
            hist = np.array([mapped_hist[1], mapped_hist[2], mapped_hist[3]], dtype=np.int64)
    return hist


def evaluate(model, loader, device, cfg):
    model.eval()

    all_contact_true, all_contact_pred = [], []
    all_mapped_true, all_mapped_pred = [], []
    all_wear_true, all_wear_pred = [], []

    total_loss = 0.0
    n_batches = 0

    contact_w = compute_class_weights_from_hist(build_contact_hist(loader.dataset)).to(device)
    mapped_w = compute_class_weights_from_hist(loader.dataset.label_histogram("mapped_class", num_classes=4)).to(device)
    wear_w = compute_class_weights_from_hist(build_wear_hist(loader.dataset)).to(device)

    lw = cfg["train"]["loss_weights"]
    use_quantization = bool(cfg["train"].get("use_quantization", True))
    quant_bits = int(cfg["train"].get("quant_bits", 8))

    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device)
            contact_y = batch["contact_label"].to(device)
            wear_y = batch["wear_label"].to(device)
            mapped_y = batch["mapped_class"].to(device)

            out = model(x, quant_bits=quant_bits, force_quantize=use_quantization)

            loss_contact = F.cross_entropy(out["contact_logits"], contact_y, weight=contact_w)
            loss_mapped = F.cross_entropy(out["mapped_logits"], mapped_y, weight=mapped_w)
            loss_wear = F.cross_entropy(out["wear_logits"], wear_y, weight=wear_w, ignore_index=-1)

            if out["reconstruction"] is not None:
                loss_recon = F.mse_loss(out["reconstruction"], x)
            else:
                loss_recon = torch.tensor(0.0, device=device)

            loss = (
                lw["contact"] * loss_contact +
                lw["wear"] * loss_wear +
                lw["mapped"] * loss_mapped +
                lw["reconstruction"] * loss_recon
            )

            total_loss += float(loss.item())
            n_batches += 1

            contact_pred = torch.argmax(out["contact_probs"], dim=1)
            mapped_pred = torch.argmax(out["mapped_probs"], dim=1)
            wear_pred = torch.argmax(out["wear_probs"], dim=1)

            all_contact_true.extend(contact_y.detach().cpu().numpy().tolist())
            all_contact_pred.extend(contact_pred.detach().cpu().numpy().tolist())
            all_mapped_true.extend(mapped_y.detach().cpu().numpy().tolist())
            all_mapped_pred.extend(mapped_pred.detach().cpu().numpy().tolist())

            valid_mask = wear_y != -1
            if valid_mask.any():
                all_wear_true.extend(wear_y[valid_mask].detach().cpu().numpy().tolist())
                all_wear_pred.extend(wear_pred[valid_mask].detach().cpu().numpy().tolist())

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "contact": macro_classification_metrics(all_contact_true, all_contact_pred, num_classes=3),
        "mapped": macro_classification_metrics(all_mapped_true, all_mapped_pred, num_classes=4),
    }
    if len(all_wear_true) > 0:
        metrics["wear"] = macro_classification_metrics(all_wear_true, all_wear_pred, num_classes=3)
    else:
        metrics["wear"] = None

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/industrial_semantic.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("train_semantic_sender")
    set_seed(cfg.get("seed", 42))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    train_ds = ToolWearDataset(
        index_csv=cfg["dataset"]["train_index"],
        root_dir=cfg["dataset"]["root_dir"],
        normalization=cfg["dataset"]["normalization"],
    )
    valid_ds = ToolWearDataset(
        index_csv=cfg["dataset"]["valid_index"],
        root_dir=cfg["dataset"]["root_dir"],
        normalization=cfg["dataset"]["normalization"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    model = SemanticSenderModel(cfg["model"]).to(device)

    contact_hist = build_contact_hist(train_ds)
    wear_hist = build_wear_hist(train_ds)
    mapped_hist = train_ds.label_histogram("mapped_class", num_classes=4)

    logger.info(f"contact_hist={contact_hist.tolist()}")
    logger.info(f"wear_hist={wear_hist.tolist()}")
    logger.info(f"mapped_hist={mapped_hist.tolist()}")

    contact_w = compute_class_weights_from_hist(contact_hist).to(device)
    wear_w = compute_class_weights_from_hist(wear_hist).to(device)
    mapped_w = compute_class_weights_from_hist(mapped_hist).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    lw = cfg["train"]["loss_weights"]
    use_quantization = bool(cfg["train"].get("use_quantization", True))
    quant_bits = int(cfg["train"].get("quant_bits", 8))

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["signal"].to(device)
            contact_y = batch["contact_label"].to(device)
            wear_y = batch["wear_label"].to(device)
            mapped_y = batch["mapped_class"].to(device)

            out = model(x, quant_bits=quant_bits, force_quantize=use_quantization)

            loss_contact = F.cross_entropy(out["contact_logits"], contact_y, weight=contact_w)
            loss_mapped = F.cross_entropy(out["mapped_logits"], mapped_y, weight=mapped_w)
            loss_wear = F.cross_entropy(out["wear_logits"], wear_y, weight=wear_w, ignore_index=-1)

            if out["reconstruction"] is not None:
                loss_recon = F.mse_loss(out["reconstruction"], x)
            else:
                loss_recon = torch.tensor(0.0, device=device)

            loss = (
                lw["contact"] * loss_contact +
                lw["wear"] * loss_wear +
                lw["mapped"] * loss_mapped +
                lw["reconstruction"] * loss_recon
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        val_metrics = evaluate(model, valid_loader, device, cfg)

        logger.info(
            f"Epoch [{epoch}/{cfg['train']['epochs']}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_mapped_acc={val_metrics['mapped']['accuracy']:.4f} "
            f"val_mapped_f1={val_metrics['mapped']['f1_macro']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "config": cfg,
            "model_state_dict": model.state_dict(),
            "val_metrics": val_metrics,
        }

        torch.save(ckpt, save_dir / "last.pt")

        score = val_metrics["mapped"]["f1_macro"]
        if score > best_score:
            best_score = score
            torch.save(ckpt, save_dir / "best.pt")
            logger.info(f"已更新 best.pt, score={best_score:.6f}")

    logger.info("训练结束。")


if __name__ == "__main__":
    main()