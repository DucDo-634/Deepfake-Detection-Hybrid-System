"""
=========================================================================================
DEEPFAKE DETECTION MODEL 
Architecture: EfficientNet-B3 Hybrid (Spatial Domain + SRM + FFT)
=========================================================================================
"""
import os, gc, io, time, math, random, logging
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.amp import autocast, GradScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── HARDWARE SETUP ────────────────────────────
torch.set_float32_matmul_precision('high') 
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

try:
    try:    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    except AttributeError: torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    torch.serialization.add_safe_globals([np.float64])
except Exception: pass

try:
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError: SKLEARN_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 ▸ SYSTEM CONFIGURATION & HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "INPUT_DIR"           : "./dataset/RealVsFake", 
    "SAVE_DIR"            : "./output",
    "IMAGE_SIZE"          : 300,       
    "BATCH_SIZE"          : 16,         
    "NUM_EPOCHS"          : 15,        
    "ACCUM_STEPS"         : 4,         
    "NUM_WORKERS"         : 6,         
    "SEED"                : 42,
    "VAL_SPLIT"           : 0.15,
    "LR_UNFREEZE"         : 4e-5,      
    "LR_MIN"              : 1e-7,
    "USE_MIXUP"           : True,
    "MIXUP_ALPHA"         : 0.2,       
    "USE_EMA"             : True,
    "EMA_DECAY"           : 0.9999,
    "USE_WEIGHTED_SAMPLER": True,
    "POS_WEIGHT_AUTO"     : False,
    "FOCAL_GAMMA"         : 2.0,
    "LABEL_SMOOTHING"     : 0.05,
    "PENALTY_THRESHOLD"   : 0.7,
    "PENALTY_FACTOR"      : 2.5,
    "BACKBONE_CHANNELS"   : 1536,      
    "SRM_DIM"             : 32,
    "FFT_DIM"             : 32,
    "USE_SAM"             : False,      
}

# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 ▸ SYSTEM UTILITIES & SAM OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) Optimizer.
    Improves model generalization by minimizing both the loss value 
    and the sharpness of the loss landscape.
    """
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"[Configuration Error] rho must be non-negative. Received: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                self.state[p]["e_w"] = e_w
                p.add_(e_w)
        if zero_grad: self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad(set_to_none=True)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack([p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group["params"] if p.grad is not None]), p=2)

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

def setup_environment(cfg):
    """Set up the computational environment, random seed, and logging system."""
    os.makedirs(cfg["SAVE_DIR"], exist_ok=True)
    random.seed(cfg["SEED"]); np.random.seed(cfg["SEED"]); torch.manual_seed(cfg["SEED"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg["SEED"])
    
    logger = logging.getLogger("Deepfake_Detection")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('[%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        fh  = logging.FileHandler(os.path.join(cfg["SAVE_DIR"], "training_log.txt"), mode='w', encoding='utf-8')
        fh.setFormatter(fmt);  logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt);  logger.addHandler(ch)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): 
        logger.info(f"[Hardware] Initializing GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"[Configuration] TF32 (TensorFloat-32) mode enabled.")
    return logger, device

# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 ▸ DATA PREPROCESSING & AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
class PadToSquare:
    """Pad image to a square aspect ratio prior to resizing to prevent distortion."""
    def __call__(self, img):
        w, h = img.size; m = max(w, h); pl, pt = (m - w) // 2, (m - h) // 2
        return ImageOps.expand(img, (pl, pt, m - w - pl, m - h - pt), fill=0)

def build_transforms(cfg, mode='train'):
    """Construct a pipeline of morphological and optical noise transformations."""
    sz, INTERP, norm = cfg["IMAGE_SIZE"], transforms.InterpolationMode.LANCZOS, transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if mode != 'train': return transforms.Compose([PadToSquare(), transforms.Resize((sz, sz), interpolation=INTERP), transforms.ToTensor(), norm])
    
    return transforms.Compose([
        PadToSquare(), 
        transforms.Resize((sz, sz), interpolation=INTERP),
        transforms.RandomHorizontalFlip(p=0.5),
        # [ACCURACY IMPROVEMENT] Reduce rotation amplitude to preserve facial structural integrity
        transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
        transforms.RandomResizedCrop(size=sz, scale=(0.90, 1.0), ratio=(0.98, 1.02), interpolation=INTERP),
        # Enhance resilience to lighting changes and realistic compression blur
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)], p=0.4),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.20),
        transforms.ToTensor(), 
        norm,
        transforms.RandomErasing(p=0.1, scale=(0.01, 0.05), ratio=(0.5, 2.0), value=0),
    ])

# ══════════════════════════════════════════════════════════════════════════════
#  PART 4 ▸ DATA MANAGEMENT (DATASET & DATALOADER)
# ══════════════════════════════════════════════════════════════════════════════
class DeepfakeDataset(Dataset):
    def __init__(self, paths, labels, transform=None, image_size=300):
        self.paths = np.array(paths, dtype=np.bytes_)
        self.labels = np.array(labels, dtype=np.float32)
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self): return len(self.paths)
    
    def __getitem__(self, idx):
        try:
            path = self.paths[idx].decode('utf-8') if isinstance(self.paths[idx], (bytes, np.bytes_)) else str(self.paths[idx])
            with open(path, 'rb') as f: 
                img = Image.open(io.BytesIO(f.read())).convert('RGB')
            if self.transform: 
                img = self.transform(img)
            return img, torch.tensor(self.labels[idx])
        except Exception as e:
            # Handle corrupted images (Silent Failure) by randomly sampling another image
            random_idx = random.randint(0, len(self.paths) - 1)
            return self.__getitem__(random_idx)

def scan_dataset(cfg, logger):
    """Recursively traverse the directory tree to index image files."""
    EXCLUDE, EXTS, paths, labels, stats = {'test', 'val', 'valid', 'validation', 'testing', 'checkpoint'}, {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}, [], [], {"REAL": 0, "FAKE": 0}
    logger.info("[Data] Starting SSD storage scanning engine...")
    for root, dirs, files in os.walk(cfg["INPUT_DIR"]):
        if 'celeb' in root.lower(): continue
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDE]
        parts = root.lower().split(os.sep)
        if any(any(exc in part for exc in EXCLUDE) for part in parts): continue
        label = 1.0 if any(('real' in p or 'original' in p) and 'fake' not in p for p in parts) else (0.0 if any(('fake' in p or 'synthesis' in p or 'synthetic' in p) and 'real' not in p for p in parts) else None)
        if label is None: continue
        for f in [f for f in files if os.path.splitext(f)[1].lower() in EXTS]:
            paths.append(os.path.join(root, f)); labels.append(label)
            stats["REAL" if label == 1.0 else "FAKE"] += 1
            
    logger.info(f"[Data] Total scanned: {stats['REAL']:,} Real images and {stats['FAKE']:,} Fake images.")
    return paths, labels, stats['REAL'], stats['FAKE']

def build_weighted_sampler(labels):
    """Balance intrinsic data distribution via Weighted Random Sampling."""
    arr = np.array(labels); n_real, n_fake = (arr == 1.0).sum(), (arr == 0.0).sum(); total = len(arr)
    weights = np.where(arr == 1.0, total / (2 * n_real) if n_real > 0 else 1.0, total / (2 * n_fake) if n_fake > 0 else 1.0)
    return WeightedRandomSampler(torch.from_numpy(weights).float(), len(weights), replacement=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PART 5 ▸ FREQUENCY AND SPATIAL DOMAIN FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
class CBAMChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__(); mid = max(channels // reduction, 8)
        self.fc_avg = nn.Sequential(nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True), nn.Linear(mid, channels, bias=False))
        self.fc_max = nn.Sequential(nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True), nn.Linear(mid, channels, bias=False))
    def forward(self, x):
        b, c = x.shape[:2]; return x * torch.sigmoid(self.fc_avg(x.mean(dim=[2, 3])) + self.fc_max(x.amax(dim=[2, 3]))).view(b, c, 1, 1)

class CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__(); self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    def forward(self, x): return x * torch.sigmoid(self.conv(torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)))

class SRMFilter(nn.Module):
    """Extract high-frequency noise artifacts via Spatial Rich Model filters (GAN/Diffusion signatures)."""
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.tensor([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]], dtype=torch.float32).unsqueeze(1).repeat(3, 1, 1, 1))
    def forward(self, x):
        with torch.no_grad(): return torch.abs(F.conv2d(x, self.weight, padding=1, groups=3)).detach()

class FFTBranch(nn.Module):
    """Analyze power spectrum anomalies via Fast Fourier Transform (FFT)."""
    def __init__(self, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(inplace=True), nn.MaxPool2d(4), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        self.proj = nn.Linear(64, out_dim)
    def forward(self, x):
        with torch.no_grad(): mag = torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))) + 1e-8)
        return self.proj(self.net(mag.float()).flatten(1))

# ══════════════════════════════════════════════════════════════════════════════
#  PART 6 ▸ HYBRID CNN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
class DeepfakeDetector(nn.Module):
    """Deepfake detection network utilizing an EfficientNet-B3 backbone with forensic features (SRM & FFT)."""
    def __init__(self, cfg, dropout=0.45):
        super().__init__()
        bb_ch, srm_dim, fft_dim = cfg["BACKBONE_CHANNELS"], cfg["SRM_DIM"], cfg["FFT_DIM"]
        self.backbone = models.efficientnet_b3(weights='DEFAULT').features
        self.cbam_ch, self.cbam_sp, self.pool_bb = CBAMChannelAttention(bb_ch, reduction=16), CBAMSpatialAttention(kernel_size=7), nn.AdaptiveAvgPool2d((1, 1))
        self.srm = SRMFilter()
        self.srm_cnn = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.SiLU(inplace=True), nn.Conv2d(16, srm_dim, 3, stride=2, padding=1), nn.BatchNorm2d(srm_dim), nn.SiLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        self.fft = FFTBranch(out_dim=fft_dim)
        
        # Dense classification head (Concatenation dimension: 1536 + 32 + 32 = 1600)
        self.classifier = nn.Sequential(
            nn.Linear(bb_ch + srm_dim + fft_dim, 512), nn.BatchNorm1d(512), nn.SiLU(inplace=True), nn.Dropout(p=dropout),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.SiLU(inplace=True), nn.Dropout(p=dropout * 0.6),
            nn.Linear(128, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize with Kaiming Normal to prevent gradient degradation during initial training phases."""
        for module in [self.srm_cnn, self.fft, self.classifier, self.cbam_ch, self.cbam_sp]:
            for m in module.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x): 
        return self.classifier(torch.cat([self.pool_bb(self.cbam_sp(self.cbam_ch(self.backbone(x)))).flatten(1), self.srm_cnn(self.srm(x)).flatten(1), self.fft(x)], dim=1))

# ══════════════════════════════════════════════════════════════════════════════
#  PART 7 ▸ OBJECTIVE FUNCTION & OPTIMIZATION UTILS
# ══════════════════════════════════════════════════════════════════════════════
class CalibratedFocalLoss(nn.Module):
    """
    Calibrated Focal Loss. Incorporates hard clamping constraints to prevent
    numerical instability (loss=nan) when computing large gradients.
    """
    def __init__(self, pos_weight=1.0, gamma=2.0, smoothing=0.05, penalty_threshold=0.7, penalty_factor=2.5):
        super().__init__(); self.pw, self.gamma, self.smooth, self.pthresh, self.pfac = pos_weight, gamma, smoothing, penalty_threshold, penalty_factor
    
    def forward(self, logits, targets):
        # Compute safe probabilities to prevent numerical explosion
        probs = torch.sigmoid(logits).clamp(min=1e-6, max=1.0 - 1e-6)
        soft  = targets * (1 - self.smooth) + 0.5 * self.smooth
        
        # Derivative of BCE with logits directly for enhanced analytical stability
        bce   = F.binary_cross_entropy_with_logits(logits, soft, reduction='none')
        err   = torch.abs(probs - targets)
        heavy = (err > self.pthresh).float() * (self.pfac - 1.0) + 1.0
        pw    = targets * (self.pw - 1.0) + 1.0
        
        loss  = bce * torch.pow(err, self.gamma) * heavy * pw
        # Ultimate safeguard: if any anomalous NaN element appears, clamp straight to 0
        return torch.where(torch.isnan(loss), torch.zeros_like(loss), loss).mean()

class EMA:
    """Exponential Moving Average to stabilize model weights over time."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in (model.module.state_dict() if hasattr(model, 'module') else model.state_dict()).items()}
        for v in self.shadow.values(): v.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for k, v in (model.module.state_dict() if hasattr(model, 'module') else model.state_dict()).items():
            if v.dtype.is_floating_point: self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)
            else: self.shadow[k].copy_(v)
    def apply(self, model): (model.module if hasattr(model, 'module') else model).load_state_dict({k: v.to(next((model.module if hasattr(model, 'module') else model).parameters()).device) for k, v in self.shadow.items()})

def mixup_batch(images, labels, alpha=0.3):
    """Random Mixup interpolation to smooth out the decision boundary."""
    if alpha <= 0: return images, labels
    lam = np.random.beta(alpha, alpha); idx = torch.randperm(images.size(0), device=images.device)
    return (lam * images + (1 - lam) * images[idx], lam * labels + (1 - lam) * labels[idx])

# ══════════════════════════════════════════════════════════════════════════════
#  PART 8 ▸ TRAINING AND EVALUATION LOOPS
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler, ema, cfg, epoch, device, logger):
    model.train(); total_loss, n_steps, accum = 0.0, len(loader), cfg["ACCUM_STEPS"]
    optimizer.zero_grad(set_to_none=True)
    
    for step, (imgs, lbls) in enumerate(loader):
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True).float().view(-1, 1)
        if cfg["USE_MIXUP"] and random.random() < 0.7: imgs, lbls = mixup_batch(imgs, lbls, cfg["MIXUP_ALPHA"])
        
        # Apply BFloat16 Mixed Precision (immune to NaN underflow) compatible with Blackwell
        with autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()): 
            loss = criterion(model(imgs), lbls) / accum
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accum == 0 or (step + 1) == n_steps:
            scaler.unscale_(optimizer if not cfg["USE_SAM"] else optimizer.base_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            if cfg["USE_SAM"]:
                optimizer.first_step(zero_grad=False)
                with autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    loss_second = criterion(model(imgs), lbls) / accum
                scaler.scale(loss_second).backward()
                optimizer.second_step(zero_grad=True)
            else:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
            # Invoke Scheduler AFTER Optimizer has successfully updated weights
            if scheduler: scheduler.step()
            if ema: ema.update(model)
            
        total_loss += loss.item() * accum
        if (step + 1) % 200 == 0 or (step + 1) == n_steps: 
            current_lr = optimizer.param_groups[0]['lr'] if not cfg["USE_SAM"] else optimizer.base_optimizer.param_groups[0]['lr']
            logger.info(f"  [Progress] Epoch {epoch+1} | Step {step+1:>5}/{n_steps} | Loss: {total_loss/(step+1):.4f} | LR: {current_lr:.2e}")
            
    return total_loss / n_steps

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model performance on the Validation set (Cross-validation)."""
    model.eval(); total_loss, all_probs, all_lbl = 0.0, [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True).float().view(-1, 1)
        with autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits = model(imgs); total_loss += criterion(logits, lbls).item()
        
        # [Type Casting Fix] Force data format to float32 for NumPy compatibility
        all_probs.extend(torch.sigmoid(logits).cpu().to(torch.float32).numpy().flatten())
        all_lbl.extend(lbls.cpu().to(torch.float32).numpy().flatten())
        
    probs, lbls = np.array(all_probs), np.array(all_lbl); preds = (probs >= 0.5).astype(float); m = {"loss": float(total_loss / len(loader)), "accuracy": float((preds == lbls).mean() * 100)}
    if SKLEARN_AVAILABLE:
        try: m.update({"auc": float(roc_auc_score(lbls, probs) * 100), "f1": float(f1_score(lbls, preds, zero_division=0) * 100)})
        except Exception: pass
    return m

def plot_training_curves(history, save_dir):
    """Visualize training convergence via Loss/AUC curve plots."""
    epochs = range(1, len(history['train_loss']) + 1); fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', lw=2)
    if history.get('val_loss'): axes[0].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', lw=2)
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss Curve'); axes[0].legend(); axes[0].grid(alpha=0.3)
    if history.get('val_acc'): axes[1].plot(epochs, history['val_acc'], 'g-^', label='Accuracy (%)', lw=2)
    if history.get('val_auc'): axes[1].plot(epochs, history['val_auc'], 'm-D', label='AUC-ROC (%)', lw=2)
    axes[1].set(xlabel='Epoch', ylabel='Score (%)', title='Classification Metrics', ylim=[40, 101]); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight'); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  PART 9 ▸ MAIN EXECUTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    logger, device = setup_environment(CONFIG)
    paths, labels, n_real, n_fake = scan_dataset(CONFIG, logger)
    if n_real == 0 or n_fake == 0: 
        logger.error("[Warning] Class distribution error! Insufficient images found in INPUT_DIR."); return None, None

    if SKLEARN_AVAILABLE: 
        tr_p, vl_p, tr_l, vl_l = train_test_split(paths, labels, test_size=CONFIG["VAL_SPLIT"], random_state=CONFIG["SEED"], stratify=labels)
    else:
        combined = list(zip(paths, labels)); random.shuffle(combined); split = int(len(paths) * (1 - CONFIG["VAL_SPLIT"]))
        tr_p, tr_l = zip(*combined[:split]); vl_p, vl_l = zip(*combined[split:])
    del paths, labels; gc.collect()

    logger.info(f"[System] Training set size: {len(tr_p):,} | Validation set: {len(vl_p):,}")

    train_ds = DeepfakeDataset(tr_p, tr_l, transform=build_transforms(CONFIG, 'train'), image_size=CONFIG["IMAGE_SIZE"])
    val_ds   = DeepfakeDataset(vl_p, vl_l, transform=build_transforms(CONFIG, 'val'), image_size=CONFIG["IMAGE_SIZE"])
    sampler = build_weighted_sampler(tr_l) if CONFIG["USE_WEIGHTED_SAMPLER"] else None

    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], sampler=sampler, shuffle=(sampler is None), num_workers=CONFIG["NUM_WORKERS"], drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"] * 2, shuffle=False, num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

    model = DeepfakeDetector(cfg=CONFIG, dropout=0.45).to(device)
    ema = EMA(model, CONFIG["EMA_DECAY"]) if CONFIG["USE_EMA"] else None
    criterion = CalibratedFocalLoss(pos_weight=(n_fake / (n_real + 1e-9)) if CONFIG["POS_WEIGHT_AUTO"] else 1.0, gamma=CONFIG["FOCAL_GAMMA"], smoothing=CONFIG["LABEL_SMOOTHING"], penalty_threshold=CONFIG["PENALTY_THRESHOLD"], penalty_factor=CONFIG["PENALTY_FACTOR"])
    
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())
    history = {k: [] for k in ['train_loss', 'val_loss', 'val_acc', 'val_auc', 'val_f1']}; best_auc, best_epoch, t_start = 0.0, 0, time.time()

    logger.info(f"[Process] Starting training pipeline for {CONFIG['NUM_EPOCHS']} epochs.")
    for p in model.parameters(): p.requires_grad = True

    if CONFIG["USE_SAM"]:
        optimizer = SAM(model.parameters(), optim.AdamW, lr=CONFIG["LR_UNFREEZE"], weight_decay=1e-5)
        base_opt = optimizer.base_optimizer
    else:
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR_UNFREEZE"], weight_decay=1e-5)
        base_opt = optimizer

    scheduler = optim.lr_scheduler.LambdaLR(base_opt, lambda step: max(CONFIG["LR_MIN"] / CONFIG["LR_UNFREEZE"], 0.5 * (1.0 + math.cos(math.pi * (step / max(1, CONFIG["NUM_EPOCHS"] * (len(train_loader) // CONFIG["ACCUM_STEPS"])))))))

    for epoch in range(CONFIG["NUM_EPOCHS"]):
        t0 = time.time(); logger.info(f"\n{'─'*60}\n  EPOCH {epoch+1}/{CONFIG['NUM_EPOCHS']}\n{'─'*60}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, scheduler, ema, CONFIG, epoch, device, logger)
        history['train_loss'].append(tr_loss)

        if ema:
            orig_cpu = {k: v.cpu() for k, v in (model.module.state_dict() if hasattr(model, 'module') else model.state_dict()).items()}
            ema.apply(model); vm = evaluate(model, val_loader, criterion, device)
            (model.module if hasattr(model, 'module') else model).load_state_dict({k: v.to(device) for k, v in orig_cpu.items()})
        else: vm = evaluate(model, val_loader, criterion, device)

        history['val_loss'].append(vm.get('loss', 0)); history['val_acc'].append(vm.get('accuracy', 0)); history['val_auc'].append(vm.get('auc', 0)); history['val_f1'].append(vm.get('f1', 0))
        logger.info(f"  [Epoch {epoch+1} Statistics] Train Loss: {tr_loss:.4f} | Val Loss: {vm.get('loss',0):.4f} | AUC: {vm.get('auc',0):.2f}% | Time: {time.time()-t0:.0f}s")

        ckpt = {'epoch': epoch + 1, 'model_state': (model.module.state_dict() if hasattr(model, 'module') else model.state_dict()), 'ema_state': ({k: v.clone() for k, v in ema.shadow.items()} if ema else None), 'val_metrics': vm, 'config': CONFIG}
        torch.save(ckpt, os.path.join(CONFIG["SAVE_DIR"], f"model_epoch_{epoch+1}.pth"))
        if vm.get('auc', 0) > best_auc:
            best_auc, best_epoch = vm.get('auc', 0), epoch + 1; torch.save(ckpt, os.path.join(CONFIG["SAVE_DIR"], "best_model.pth"))
            logger.info(f"  [Storage] Saving new optimal checkpoint: AUC = {best_auc:.2f}%")

    logger.info("\n" + "=" * 72 + f"\n  [CONCLUSION] TRAINING COMPLETE | Best Epoch: {best_epoch} | Peak AUC: {best_auc:.2f}%\n" + "=" * 72)
    plot_training_curves(history, CONFIG["SAVE_DIR"])
    return model, history

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()