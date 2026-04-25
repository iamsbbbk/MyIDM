import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, utils as vutils
from tqdm import tqdm
from datetime import datetime
import traceback
from torch.utils.tensorboard import SummaryWriter

from diffusers import UNet2DConditionModel
from model import Net
from skimage.metrics import peak_signal_noise_ratio as psnr_calc
from skimage.metrics import structural_similarity as ssim_calc

# ==========================================
# 0. 核心创新模块 (FFL + 可学习采样矩阵)
# ==========================================
class FocalFrequencyLoss(nn.Module):
    """频域损失函数：惩罚高频细节恢复的误差"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='none')

    def forward(self, pred, target):
        pred_freq = torch.fft.fftn(pred, dim=(-2, -1))
        target_freq = torch.fft.fftn(target, dim=(-2, -1))
        
        pred_freq = torch.stack([pred_freq.real, pred_freq.imag], -1)
        target_freq = torch.stack([target_freq.real, target_freq.imag], -1)

        freq_distance = self.l1_loss(pred_freq, target_freq).mean(dim=-1)
        weight = (freq_distance ** self.alpha)
        weight = weight / (weight.max() + 1e-8)
        
        return (freq_distance * weight).mean()

class LearnableCSMatrix(nn.Module):
    """端到端可学习的压缩感知测量矩阵"""
    def __init__(self, N, q, device):
        super().__init__()
        # 初始依然保持正交性，帮助稳定训练
        U, S, V = torch.linalg.svd(torch.randn(N, N, device=device))
        self.Phi = nn.Parameter((U @ V)[:, :q].float())
    
    def forward(self):
        return self.Phi

# ==========================================
# 1. Dataset (鲁棒最大最小平均化)
# ==========================================
class ToolWearExpertDataset(Dataset):
    def __init__(self, root_dir, target_class, split='train'):
        self.data = []
        file_path = os.path.join(root_dir, f"rgb_x_{target_class}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        try:
            all_data = np.load(file_path, mmap_mode='r')
            all_data = np.array(all_data).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load numpy file: {e}")

        # ---------------------------------------------------------
        # 💡 [工程优化]: 鲁棒的最大最小平均化 (Robust Min-Max)
        # 保证数值不会太小，剔除极端的反光亮点或黑点干扰
        # ---------------------------------------------------------
        p_min = np.percentile(all_data, 1)   # 取 1% 分位数
        p_max = np.percentile(all_data, 99)  # 取 99% 分位数
        
        # 1. 截断极端值
        all_data = np.clip(all_data, p_min, p_max)
        # 2. 最大最小归一化
        all_data = (all_data - p_min) / (p_max - p_min + 1e-8)

        np.random.seed(42)
        indices = np.random.permutation(len(all_data))
        all_data = all_data[indices]

        split_idx = int(len(all_data) * 0.85)

        if split == 'train':
            self.data = all_data[:split_idx]
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            self.data = all_data[split_idx:]
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img

# ==========================================
# 2. Operator Helper
# ==========================================
def make_A_AT_for_patch(im_size, block_size, Phi):
    # 移除 unfold/fold 的重复初始化，加快速度
    unfold = nn.Unfold(kernel_size=block_size, stride=block_size)
    fold = nn.Fold(output_size=(im_size, im_size), kernel_size=block_size, stride=block_size)

    def A(x):
        return torch.matmul(Phi.t(), unfold(x))

    def AT(y):
        return fold(torch.matmul(Phi, y))

    return A, AT

# ==========================================
# 3. Loss & Metrics
# ==========================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

def calculate_metrics(img1, img2):
    img1 = img1.detach().cpu().numpy().transpose(0, 2, 3, 1)
    img2 = img2.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    psnr_val, ssim_val = 0.0, 0.0
    batch_size = img1.shape[0]
    
    for i in range(batch_size):
        psnr_val += psnr_calc(img2[i], img1[i], data_range=1.0)
        ssim_val += ssim_calc(img2[i], img1[i], data_range=1.0, channel_axis=-1, win_size=3)
        
    return psnr_val / batch_size, ssim_val / batch_size

# ==========================================
# 4. Main Training Logic
# ==========================================
def setup_logging(save_dir, exp_name):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        sh = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

def save_validation_images(x_gt, x_rec, epoch, save_dir):
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    comparison = torch.cat([x_gt[:8], x_rec[:8]], dim=0)
    grid = vutils.make_grid(comparison, nrow=8, padding=2, normalize=True)
    vutils.save_image(grid, os.path.join(vis_dir, f"epoch_{epoch}_compare.png"))

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"./data/ToolWear_RGB")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_idm_phys")
    parser.add_argument("--sd_path", type=str, default="./sd15")
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step_number", type=int, default=8)
    parser.add_argument("--cs_ratio", type=float, default=0.1)
    parser.add_argument("--block_size", type=int, default=8)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = f"EagleEye_C{args.target_class}_R{args.cs_ratio}"
    logger = setup_logging(args.save_dir, exp_name)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs", exp_name))
    logger.info(f"Config: {vars(args)}")

    # 1. Load Model
    try:
        unet = UNet2DConditionModel.from_pretrained(args.sd_path, subfolder="unet", local_files_only=True)
    except Exception:
        unet = UNet2DConditionModel.from_pretrained(args.sd_path, local_files_only=True)

    model = Net(T=args.step_number, unet=unet).to(device)

    # 2. CS Matrix Setup (可学习矩阵)
    N = args.block_size ** 2
    q = int(np.ceil(args.cs_ratio * N))
    cs_matrix_module = LearnableCSMatrix(N, q, device).to(device)
    logger.info(f"Learnable CS Matrix: N={N}, q={q}, Ratio={args.cs_ratio}")

    # 3. Optimizer (联合优化网络和矩阵)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': cs_matrix_module.parameters(), 'lr': args.lr * 0.1} # 矩阵的学习率稍小，保持物理稳定性
    ], weight_decay=1e-2)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

    # 4. Losses
    criterion_img = CharbonnierLoss().to(device)
    criterion_meas = CharbonnierLoss().to(device)
    criterion_freq = FocalFrequencyLoss().to(device)
    
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # 5. Data
    train_dataset = ToolWearExpertDataset(args.data_dir, args.target_class, 'train')
    val_dataset = ToolWearExpertDataset(args.data_dir, args.target_class, 'val')
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    best_psnr = 0.0

    # 6. Training Loop
    for epoch in range(1, args.epoch + 1):
        model.train()
        cs_matrix_module.train()
        loss_avg = 0.0
        
        # 频域损失 Warm-up 策略
        # 前 5 个 Epoch 专注学轮廓，第 6 个 Epoch 开始逐渐加入高频惩罚
        ffl_weight = 0.0 if epoch <= 5 else min(0.1, 0.02 * (epoch - 5))

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epoch} (FFL Wt: {ffl_weight:.3f})")
        
        for step, x_gt in enumerate(pbar):
            try:
                x_gt = x_gt.to(device)
                b, c, h, w = x_gt.shape 
                x_flat = x_gt.view(b*c, 1, h, w) 
                
                # 动态获取当前的采样矩阵并生成算子
                Phi = cs_matrix_module()
                A_func, AT_func = make_A_AT_for_patch(h, args.block_size, Phi)
                
                with torch.no_grad():
                    # 模拟压缩感知测量
                    y = A_func(x_flat)

                optimizer.zero_grad()
                
                with autocast(enabled=torch.cuda.is_available()):
                    # 模型重构 (使用我们上一版的 Attentive mHC Model)
                    x_rec_flat = model(y, A_func, AT_func)

                    # 计算损失
                    loss_img = criterion_img(x_rec_flat, x_flat)
                    loss_meas = criterion_meas(A_func(x_rec_flat), y)
                    
                    loss = loss_img + 1.0 * loss_meas
                    
                    # 激活 FFL
                    loss_freq = torch.tensor(0.0).to(device)
                    if ffl_weight > 0:
                        loss_freq = criterion_freq(x_rec_flat, x_flat)
                        loss += ffl_weight * loss_freq

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                loss_avg += loss.item()
                pbar.set_postfix({
                    'L_img': f"{loss_img.item():.4f}", 
                    'L_frq': f"{loss_freq.item():.4f}" if ffl_weight > 0 else "0.00"
                })
                
                global_step = (epoch - 1) * len(train_loader) + step
                if step % 10 == 0:
                    writer.add_scalar('Loss/Total', loss.item(), global_step)
                    writer.add_scalar('Loss/Image', loss_img.item(), global_step)
                    if ffl_weight > 0:
                        writer.add_scalar('Loss/Freq', loss_freq.item(), global_step)

            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                traceback.print_exc()
                continue

        scheduler.step()
        writer.add_scalar('LR/Model', optimizer.param_groups[0]['lr'], epoch)

        # 7. Validation Loop
        if epoch % 1 == 0:
            model.eval()
            cs_matrix_module.eval()
            psnr_acc, ssim_acc = 0.0, 0.0
            vis_gt, vis_rec = None, None

            with torch.no_grad():
                # 验证时使用当前学到的 Phi
                Phi = cs_matrix_module()
                
                for i, x_val in enumerate(val_loader):
                    x_val = x_val.to(device)
                    b, c, h, w = x_val.shape
                    x_flat = x_val.view(b*c, 1, h, w)
                    
                    A_func, AT_func = make_A_AT_for_patch(h, args.block_size, Phi)
                    y = A_func(x_flat)

                    x_rec_flat = model(y, A_func, AT_func)

                    x_rec_rgb = x_rec_flat.view(b, c, h, w)
                    x_rec_rgb = torch.clamp(x_rec_rgb, 0, 1)

                    p, s = calculate_metrics(x_rec_rgb, x_val)
                    psnr_acc += p
                    ssim_acc += s
                    
                    if i == 0:
                        vis_gt = x_val
                        vis_rec = x_rec_rgb

            avg_psnr = psnr_acc / len(val_loader)
            avg_ssim = ssim_acc / len(val_loader)
            avg_loss = loss_avg / len(train_loader)

            logger.info(f"Ep {epoch} | Loss: {avg_loss:.4f} | Val PSNR: {avg_psnr:.2f} dB | Val SSIM: {avg_ssim:.4f}")
            
            writer.add_scalar('Metrics/PSNR', avg_psnr, epoch)
            writer.add_scalar('Metrics/SSIM', avg_ssim, epoch)
            
            save_validation_images(vis_gt, vis_rec, epoch, args.save_dir)

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                # 同时保存模型和采样矩阵
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'matrix_state_dict': cs_matrix_module.state_dict(),
                    'config': {
                        'step_number': args.step_number,
                        'block_size': args.block_size,
                        'cs_ratio': args.cs_ratio,
                        'target_class': args.target_class,
                        'sd_path': args.sd_path,
                    }
                }, os.path.join(args.save_dir, f"best_model_cls{args.target_class}.pth"))
                logger.info(f"==> Best Saved! ({best_psnr:.2f} dB)")
    
    writer.close()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print("\n" + "!"*60)
        print("CRITICAL ERROR IN TRAINING")
        print("!"*60)
        traceback.print_exc()
        print("!"*60)
