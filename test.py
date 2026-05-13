import os
import csv
import argparse
import traceback

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from tqdm import tqdm

from diffusers import UNet2DConditionModel
from model import Net

from train import (
    set_seed,
    ToolWearExpertDataset,
    LearnableCSMatrix,
    make_A_AT_for_patch,
    calculate_metrics,
    call_model
)


def strip_module_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    return new_state


def load_unet_for_test(sd_path):
    try:
        unet = UNet2DConditionModel.from_pretrained(
            sd_path,
            subfolder="unet",
            local_files_only=True
        )
        print("[INFO] UNet loaded from sd_path/unet")
    except Exception:
        unet = UNet2DConditionModel.from_pretrained(
            sd_path,
            local_files_only=True
        )
        print("[INFO] UNet loaded from sd_path root")

    return unet


def save_batch_visualization(x_gt, x_rec, save_path, max_num=8):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x_gt = x_gt[:max_num].detach().cpu().clamp(0, 1)
    x_rec = x_rec[:max_num].detach().cpu().clamp(0, 1)
    x_err = torch.abs(x_gt - x_rec).mul(4.0).clamp(0, 1)

    comparison = torch.cat([x_gt, x_rec, x_err], dim=0)

    grid = vutils.make_grid(
        comparison,
        nrow=x_gt.shape[0],
        padding=2,
        normalize=False
    )

    vutils.save_image(grid, save_path)


@torch.no_grad()
def test():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_idm_phys")

    parser.add_argument("--data_dir", type=str, default="./data/ToolWear_RGB")
    parser.add_argument("--result_dir", type=str, default="./results_idm_phys")

    parser.add_argument("--target_class", type=int, default=None)
    parser.add_argument("--sd_path", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "all"])
    parser.add_argument("--split_ratio", type=float, default=None)

    parser.add_argument("--norm", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_vis", type=int, default=32)
    parser.add_argument("--save_npy", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------
    # 1. Resolve checkpoint path
    # -----------------------------------------------------
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        cls = 0 if args.target_class is None else args.target_class
        checkpoint_path = os.path.join(
            args.save_dir,
            f"class{cls}",
            f"best_model_cls{cls}.pth"
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 80)
    print("[Test] MyIDM on ToolWear_RGB")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device    : {device}")
    print("=" * 80)

    ckpt = torch.load(checkpoint_path, map_location=device)

    config = ckpt.get("config", {})
    norm_stats = ckpt.get("norm_stats", None)

    target_class = (
        args.target_class
        if args.target_class is not None
        else int(config.get("target_class", 0))
    )

    step_number = int(config.get("step_number", 8))
    block_size = int(config.get("block_size", 32))
    cs_ratio = float(config.get("cs_ratio", 0.1))
    sd_path = args.sd_path if args.sd_path else config.get("sd_path", "./sd15")
    split_ratio = args.split_ratio if args.split_ratio is not None else float(config.get("split_ratio", 0.85))
    norm = args.norm if args.norm else config.get("norm", "sym_p99")

    print(f"target_class : {target_class}")
    print(f"step_number  : {step_number}")
    print(f"block_size   : {block_size}")
    print(f"cs_ratio     : {cs_ratio}")
    print(f"sd_path      : {sd_path}")
    print(f"norm         : {norm}")
    print(f"split        : {args.split}")
    print(f"split_ratio  : {split_ratio}")

    # -----------------------------------------------------
    # 2. Dataset
    # -----------------------------------------------------
    test_dataset = ToolWearExpertDataset(
        root_dir=args.data_dir,
        target_class=target_class,
        split=args.split,
        split_ratio=split_ratio,
        seed=args.seed,
        norm=norm,
        stats=norm_stats,
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0)
    )

    print(test_dataset.info_string())
    print(f"Test samples: {len(test_dataset)}")

    # -----------------------------------------------------
    # 3. Model
    # -----------------------------------------------------
    unet = load_unet_for_test(sd_path)
    model = Net(T=step_number, unet=unet).to(device)

    model_state = strip_module_prefix(ckpt["model_state_dict"])
    model.load_state_dict(model_state, strict=True)
    model.eval()

    # -----------------------------------------------------
    # 4. CS Matrix
    # -----------------------------------------------------
    matrix_state = ckpt["matrix_state_dict"]
    phi_saved = matrix_state["Phi"]

    N_saved, q_saved = phi_saved.shape

    N_expected = block_size ** 2
    if N_saved != N_expected:
        raise ValueError(
            f"Checkpoint Phi N={N_saved}, but block_size^2={N_expected}. "
            f"Please check checkpoint/config."
        )

    cs_matrix_module = LearnableCSMatrix(
        N=N_saved,
        q=q_saved,
        device=device
    ).to(device)

    cs_matrix_module.load_state_dict(matrix_state, strict=True)
    cs_matrix_module.eval()

    Phi = cs_matrix_module()

    print(f"Loaded Phi: shape={tuple(Phi.shape)}")

    # -----------------------------------------------------
    # 5. Result directories
    # -----------------------------------------------------
    result_dir = os.path.join(
        args.result_dir,
        f"class{target_class}",
        f"R{cs_ratio}_B{block_size}_T{step_number}"
    )

    vis_dir = os.path.join(result_dir, "visualizations")
    npy_dir = os.path.join(result_dir, "npy")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    if args.save_npy:
        os.makedirs(npy_dir, exist_ok=True)

    # -----------------------------------------------------
    # 6. Testing Loop
    # -----------------------------------------------------
    total_psnr = 0.0
    total_ssim = 0.0
    total_num = 0

    metric_rows = []
    saved_vis = 0
    global_index = 0

    for batch_idx, x_gt in enumerate(tqdm(test_loader, desc="Testing", ncols=120)):
        x_gt = x_gt.to(device, non_blocking=True).float()

        b, c, h, w = x_gt.shape
        x_flat = x_gt.reshape(b * c, 1, h, w)

        A_func, AT_func = make_A_AT_for_patch(
            h=h,
            w=w,
            block_size=block_size,
            Phi=Phi
        )

        y = A_func(x_flat)

        with autocast(enabled=(device.type == "cuda")):
            x_rec_flat = call_model(
                model,
                y,
                A_func,
                AT_func,
                use_amp=(device.type == "cuda")
            )

        x_rec_flat = x_rec_flat[..., :h, :w]
        x_rec = x_rec_flat.reshape(b, c, h, w).clamp(0, 1)

        psnr_sum, ssim_sum, n = calculate_metrics(x_rec, x_gt)

        batch_psnr = psnr_sum / max(n, 1)
        batch_ssim = ssim_sum / max(n, 1)
        batch_mse = torch.mean((x_rec - x_gt) ** 2).item()

        total_psnr += psnr_sum
        total_ssim += ssim_sum
        total_num += n

        metric_rows.append({
            "batch": batch_idx,
            "num": n,
            "mse": batch_mse,
            "psnr": batch_psnr,
            "ssim": batch_ssim,
        })

        if saved_vis < args.save_vis:
            remain = args.save_vis - saved_vis
            cur_num = min(remain, b)

            save_path = os.path.join(
                vis_dir,
                f"batch_{batch_idx:04d}_gt_rec_err.png"
            )

            save_batch_visualization(
                x_gt[:cur_num],
                x_rec[:cur_num],
                save_path,
                max_num=cur_num
            )

            saved_vis += cur_num

        if args.save_npy:
            for i in range(b):
                np.save(
                    os.path.join(npy_dir, f"rec_{global_index:06d}.npy"),
                    x_rec[i].detach().cpu().numpy()
                )
                global_index += 1
        else:
            global_index += b

    avg_psnr = total_psnr / max(total_num, 1)
    avg_ssim = total_ssim / max(total_num, 1)

    # -----------------------------------------------------
    # 7. Save Metrics
    # -----------------------------------------------------
    csv_path = os.path.join(result_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["batch", "num", "mse", "psnr", "ssim"]
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    summary_path = os.path.join(result_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {checkpoint_path}\n")
        f.write(f"data_dir: {args.data_dir}\n")
        f.write(f"target_class: {target_class}\n")
        f.write(f"split: {args.split}\n")
        f.write(f"split_ratio: {split_ratio}\n")
        f.write(f"step_number: {step_number}\n")
        f.write(f"block_size: {block_size}\n")
        f.write(f"cs_ratio: {cs_ratio}\n")
        f.write(f"norm: {norm}\n")
        f.write(f"Phi shape: {tuple(Phi.shape)}\n")
        f.write(f"num_samples: {total_num}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.6f}\n")

    print("\n" + "=" * 80)
    print("Test Finished")
    print("=" * 80)
    print(f"Samples      : {total_num}")
    print(f"Average PSNR : {avg_psnr:.4f} dB")
    print(f"Average SSIM : {avg_ssim:.6f}")
    print(f"CSV          : {csv_path}")
    print(f"Summary      : {summary_path}")
    print(f"Vis Dir      : {vis_dir}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test()
    except Exception:
        print("\n" + "!" * 80)
        print("CRITICAL ERROR IN TESTING")
        print("!" * 80)
        traceback.print_exc()
        print("!" * 80)