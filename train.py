import os
import sys
import argparse
import logging
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# 假设 model.py 和 utils.py 在同一目录下
try:
    from model import Net
    from diffusers import StableDiffusionPipeline
except ImportError as e:
    print(f"Error: 缺少必要依赖文件。请确保 model.py 和 diffusers 库已正确安装。\n详细信息: {e}")
    sys.exit(1)


# ==========================================
# 1. 配置与日志系统 (Robust Logging)
# ==========================================
def setup_logging(save_dir, exp_name):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ==========================================
# 2. 压缩感知算子 (CS Operators)
# ==========================================
def make_A_AT_for_patch(psz, B, Phi, device, perm=None, perm_inv=None):
    """
    构建基于块的压缩感知算子 A (采样) 和 AT (重构)。
    针对 32x32 小图像，建议 Block Size 设为 8 或 4。
    """
    nb = (psz // B) * (psz // B)
    if perm is None:
        perm = torch.arange(nb, device=device)
    if perm_inv is None:
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(nb, device=device)

    # Phi shape: [B*B, q]

    def A(z):
        # z: [bsz, C, psz, psz] -> 这里的 C 通常是 1 或 3，CS 通常针对单通道处理或逐通道
        # 为了简化，我们假设 z 是 [bsz, 1, psz, psz] 或者我们在外部处理通道
        if z.shape[1] == 3:
            # 如果是 RGB，我们需要对每个通道分别做，或者把通道展平
            # 这里简化处理：将 RGB 视为 Batch 的一部分进行投影，或者只处理单通道
            # 方案：Reshape [B*3, 1, H, W]
            b, c, h, w = z.shape
            z = z.reshape(b * c, 1, h, w)

        z = z.float()
        blocks = F.unfold(z, kernel_size=B, stride=B)  # [bsz*c, B*B, nb]
        blocks = blocks.transpose(1, 2)  # [bsz*c, nb, B*B]
        blocks = blocks[:, perm, :]
        y_meas = blocks @ Phi  # [bsz*c, nb, q]
        return y_meas

    def AT(y):
        # y: [bsz*c, nb, q]
        y = y.float()
        blocks = y @ Phi.t()  # [bsz*c, nb, B*B]
        blocks = blocks[:, perm_inv, :]
        blocks = blocks.transpose(1, 2)  # [bsz*c, B*B, nb]
        rec = F.fold(blocks, output_size=(psz, psz), kernel_size=B, stride=B)

        # 恢复 RGB 维度
        if rec.shape[0] % 3 == 0:  # 简单的 heuristic，实际应传入原始 batch size
            # 注意：这里在 forward 中通常无法得知原始 batch size，
            # 但 IDM 的 model.py 通常处理的是 [B, C, H, W]，所以这里返回 unfolded 即可
            # 由 Model 内部处理维度
            pass
        return rec

    return A, AT


# ==========================================
# 3. 专家数据集 (Expert Dataset with Normalization)
# ==========================================
class ToolWearExpertDataset(Dataset):
    def __init__(self, data_root, target_class, mode='train', seed=42):
        """
        Args:
            data_root: 数据根目录
            target_class: 目标类别 {0, 1, 2, 3}
            mode: 'train' (70%), 'val' (20%), 'test' (10%)
        """
        self.mode = mode
        filename = f"rgb_x_{target_class}.npy"
        file_path = os.path.join(data_root, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到数据集文件: {file_path}")

        logging.info(f"正在加载专家数据集 (Class {target_class}): {file_path}")
        try:
            # 加载数据 (N, 3, 32, 32)
            all_data = np.load(file_path)
            total_samples = len(all_data)
            logging.info(f"原始数据形状: {all_data.shape}, 样本数: {total_samples}")

            # 随机打乱索引 (固定种子确保可复现)
            indices = np.arange(total_samples)
            np.random.seed(seed)
            np.random.shuffle(indices)

            # 7:2:1 划分
            n_train = int(total_samples * 0.7)
            n_val = int(total_samples * 0.2)

            if mode == 'train':
                self.indices = indices[:n_train]
            elif mode == 'val':
                self.indices = indices[n_train: n_train + n_val]
            elif mode == 'test':
                self.indices = indices[n_train + n_val:]
            else:
                raise ValueError("Mode must be 'train', 'val', or 'test'")

            self.data = all_data[self.indices]
            logging.info(f"Mode [{mode}] 加载完成: {len(self.data)} 样本")

        except Exception as e:
            logging.error(f"数据加载失败: {e}")
            raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # shape: (3, 32, 32)
        x = self.data[idx]

        # 转为 float32
        x = x.astype(np.float32)

        # ==========================================
        # 核心功能：Min-Max Normalization (Instance Level)
        # ==========================================
        # 针对每个样本单独归一化，拉伸对比度
        x_min = x.min()
        x_max = x.max()

        if x_max - x_min > 1e-8:
            x = (x - x_min) / (x_max - x_min)
        else:
            # 如果图像是纯色的（极少见），保持原样或置零
            x = x - x_min

            # 转换为 Tensor
        x_tensor = torch.from_numpy(x)

        return x_tensor


# ==========================================
# 4. 指标计算工具
# ==========================================
def calculate_metrics(pred_batch, target_batch):
    """
    计算 batch 的平均 PSNR 和 SSIM
    Input: [B, C, H, W] in range [0, 1]
    """
    psnr_vals = []
    ssim_vals = []

    # 移至 CPU 并转为 numpy
    preds = pred_batch.detach().cpu().numpy()
    targets = target_batch.detach().cpu().numpy()

    for i in range(preds.shape[0]):
        p = preds[i].transpose(1, 2, 0)  # [H, W, C]
        t = targets[i].transpose(1, 2, 0)

        # 确保范围在 [0, 1] 之间，防止数值误差
        p = np.clip(p, 0, 1)
        t = np.clip(t, 0, 1)

        psnr_vals.append(psnr_metric(t, p, data_range=1.0))
        # SSIM 需要指定 data_range 和 channel_axis
        ssim_vals.append(ssim_metric(t, p, data_range=1.0, channel_axis=2))

    return np.mean(psnr_vals), np.mean(ssim_vals)


# ==========================================
# 5. 主训练循环
# ==========================================
def train():
    parser = argparse.ArgumentParser(description="M-IDM Expert Training")
    # 路径参数
    parser.add_argument("--data_dir", type=str, default=r"D:\WORK\GraduationProgramme\IDM\MyIDM\data\ToolWear_RGB")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--sd_path", type=str, default="./sd15", help="Stable Diffusion v1.5 路径")

    # 训练参数
    parser.add_argument("--target_class", type=int, default=3, choices=[0, 1, 2, 3], help="指定训练哪个磨损状态")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)  # 图片很小(32x32)，Batch size 可以大一点
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--step_number", type=int, default=3, help="IDM 时间步数 T")
    parser.add_argument("--cs_ratio", type=float, default=0.1, help="压缩采样率")
    parser.add_argument("--block_size", type=int, default=8, help="CS 块大小，对于 32x32 图片建议设为 8")

    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化日志
    exp_name = f"Class{args.target_class}_Ratio{args.cs_ratio}"
    logger = setup_logging(args.save_dir, exp_name)
    logger.info(f"启动训练任务: {exp_name}")
    logger.info(f"参数列表: {vars(args)}")

    try:
        # --------------------
        # 1. 准备数据
        # --------------------
        train_dataset = ToolWearExpertDataset(args.data_dir, args.target_class, mode='train')
        val_dataset = ToolWearExpertDataset(args.data_dir, args.target_class, mode='val')

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # --------------------
        # 2. 准备模型 (M-IDM)
        # --------------------
        logger.info("加载 Stable Diffusion UNet...")
        # 注意：SD v1.5 默认针对 512x512。对于 32x32 输入，Latent 将变为 4x4。
        # 如果 Net 实现中能处理小尺寸则没问题，否则可能需要 Upsample 输入或调整 UNet 配置。
        # 这里假设 Net (M-IDM) 已经适配或 SD UNet 足够鲁棒。
        pipe = StableDiffusionPipeline.from_pretrained(args.sd_path, local_files_only=True, safety_checker=None).to(
            device)
        unet = pipe.unet
        unet.requires_grad_(True)  # 允许微调

        model = Net(args.step_number, unet).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scaler = GradScaler()

        # --------------------
        # 3. 准备 CS 算子
        # --------------------
        img_size = 32  # 固定为数据集尺寸
        B = args.block_size
        N = B * B
        q = int(np.ceil(args.cs_ratio * N))

        # 初始化高斯随机矩阵并正交化
        torch.manual_seed(42)  # 保证 A 矩阵在多次实验中一致
        U_mat, _, _ = torch.linalg.svd(torch.randn(N, N, device=device))
        Phi = U_mat[:, :q].float()
        logger.info(f"CS 矩阵初始化完成: Block={B}, Input={N}, Output={q}, Ratio={args.cs_ratio}")

        # --------------------
        # 4. 训练循环
        # --------------------
        best_ssim = 0.0

        for epoch in range(1, args.epoch + 1):
            model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epoch} [Train]")
            for x in pbar:
                x = x.to(device)  # [B, 3, 32, 32]

                # 由于 CS 通常处理单通道或展平，这里我们需要适配 A/AT
                # 将 RGB 视为 Batch 维度堆叠: [B*3, 1, 32, 32]
                b, c, h, w = x.shape
                x_reshaped = x.view(b * c, 1, h, w)

                # 生成动态的 A, AT (虽然 Phi 固定，但 Permutation 可以随机以增强鲁棒性，这里为了稳定暂不随机 Perm)
                A, AT = make_A_AT_for_patch(img_size, B, Phi, device)

                # CS 采样
                y = A(x_reshaped)

                optimizer.zero_grad()
                with autocast():
                    # 模型前向
                    # model 的输入逻辑需要根据 model.py 实际情况。
                    # 通常 IDM 接受 [B, C, H, W] 和对应的 A, AT
                    # 这里我们将 reshaping 还原，让 model 内部处理，或者让 model 处理 reshaped
                    # 假设 model 处理 [Batch_Any, 1, H, W]
                    x_rec_reshaped = model(y, A, AT)

                    # Loss: L1 Loss
                    loss = F.l1_loss(x_rec_reshaped, x_reshaped)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                pbar.set_postfix({'L1': f"{loss.item():.4f}"})

            avg_train_loss = train_loss / len(train_loader)

            # --------------------
            # 5. 验证循环 (Validation)
            # --------------------
            if epoch % 1 == 0:  # 每个 epoch 都验证
                model.eval()
                val_psnr = 0.0
                val_ssim = 0.0

                with torch.no_grad():
                    # val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
                    for val_x in val_loader:
                        val_x = val_x.to(device)
                        b, c, h, w = val_x.shape
                        val_x_reshaped = val_x.view(b * c, 1, h, w)

                        A_val, AT_val = make_A_AT_for_patch(img_size, B, Phi, device)
                        y_val = A_val(val_x_reshaped)

                        rec_reshaped = model(y_val, A_val, AT_val)

                        # 还原回 RGB [B, 3, H, W] 用于计算指标
                        rec_rgb = rec_reshaped.view(b, c, h, w)
                        val_x_rgb = val_x  # 原始 RGB

                        # 截断到 [0, 1]
                        rec_rgb = torch.clamp(rec_rgb, 0, 1)

                        batch_psnr, batch_ssim = calculate_metrics(rec_rgb, val_x_rgb)
                        val_psnr += batch_psnr
                        val_ssim += batch_ssim

                avg_val_psnr = val_psnr / len(val_loader)
                avg_val_ssim = val_ssim / len(val_loader)

                logger.info(
                    f"Epoch {epoch} Summary: Train Loss={avg_train_loss:.6f} | Val PSNR={avg_val_psnr:.2f} dB | Val SSIM={avg_val_ssim:.4f}")

                # 保存最佳模型
                if avg_val_ssim > best_ssim:
                    best_ssim = avg_val_ssim
                    save_path = os.path.join(args.save_dir, f"best_model_class{args.target_class}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_ssim': best_ssim,
                        'args': vars(args)
                    }, save_path)
                    logger.info(f"★ New Best Model Saved! SSIM: {best_ssim:.4f}")

    except Exception as e:
        logger.error("训练过程中发生严重错误！")
        logger.error(e, exc_info=True)
        raise e


if __name__ == "__main__":
    train()