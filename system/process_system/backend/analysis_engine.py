import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import ImageOps
import logging
import traceback

logger = logging.getLogger("ProcessingLayer.Engine")

# --- MODEL CONFIGURATION  ---
CONFIG = {
    "IMAGE_SIZE": 300,
    "BACKBONE_CHANNELS": 1536,
    "SRM_DIM": 32,
    "FFT_DIM": 32,
}

class PadToSquare:
    """Pad image to a square aspect ratio to prevent distortion."""
    def __call__(self, img):
        w, h = img.size; m = max(w, h); pl, pt = (m - w) // 2, (m - h) // 2
        return ImageOps.expand(img, (pl, pt, m - w - pl, m - h - pt), fill=0)

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
    """Extract high-frequency noise artifacts (GAN/Diffusion signatures)."""
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.tensor([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]], dtype=torch.float32).unsqueeze(1).repeat(3, 1, 1, 1))
    def forward(self, x):
        with torch.no_grad(): return torch.abs(F.conv2d(x, self.weight, padding=1, groups=3)).detach()

class FFTBranch(nn.Module):
    """Analyze power spectrum anomalies via Fast Fourier Transform."""
    def __init__(self, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU(inplace=True), nn.MaxPool2d(4), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        self.proj = nn.Linear(64, out_dim)
    def forward(self, x):
        with torch.no_grad(): mag = torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))) + 1e-8)
        return self.proj(self.net(mag.float()).flatten(1))

class DeepfakeDetector(nn.Module):
    def __init__(self, cfg, dropout=0.45):
        super().__init__()
        bb_ch, srm_dim, fft_dim = cfg["BACKBONE_CHANNELS"], cfg["SRM_DIM"], cfg["FFT_DIM"]
        self.backbone = models.efficientnet_b3(weights=None).features
        self.cbam_ch, self.cbam_sp, self.pool_bb = CBAMChannelAttention(bb_ch), CBAMSpatialAttention(kernel_size=7), nn.AdaptiveAvgPool2d((1, 1))
        self.srm = SRMFilter()
        self.srm_cnn = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.SiLU(inplace=True), nn.Conv2d(16, srm_dim, 3, stride=2, padding=1), nn.BatchNorm2d(srm_dim), nn.SiLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        self.fft = FFTBranch(out_dim=fft_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(bb_ch + srm_dim + fft_dim, 512), nn.BatchNorm1d(512), nn.SiLU(inplace=True), nn.Dropout(p=dropout),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.SiLU(inplace=True), nn.Dropout(p=dropout * 0.6),
            nn.Linear(128, 1)
        )

    def forward(self, x): 
        return self.classifier(torch.cat([self.pool_bb(self.cbam_sp(self.cbam_ch(self.backbone(x)))).flatten(1), self.srm_cnn(self.srm(x)).flatten(1), self.fft(x)], dim=1))

class LocalAnalysisEngine:
    def __init__(self, device):
        self.device = device
        
        # --- DYNAMIC PATH RESOLUTION ---
        # Traverse upwards from current file to find the 'output' directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = current_dir
        while not os.path.isdir(os.path.join(root_dir, 'output')):
            parent = os.path.dirname(root_dir)
            if parent == root_dir:
                raise FileNotFoundError("Could not locate the 'output' directory in the project tree.")
            root_dir = parent
            
        self.path_model = os.path.join(root_dir, "output", "best_model.pth")
        
        logger.info("================================================================")
        logger.info(" INITIALIZING V5 HYBRID ENGINE (EFFICIENTNET-B3 + BFLOAT16) ")
        logger.info("================================================================")
        logger.info(f"[System] Resolved weights path: {self.path_model}")

        self.model = DeepfakeDetector(cfg=CONFIG).to(self.device)
        
        # Load weights, prioritizing EMA state for optimal generalization
        checkpoint = torch.load(self.path_model, map_location=self.device)
        state_dict = checkpoint.get('ema_state') if checkpoint.get('ema_state') else checkpoint.get('model_state')
        
        # Remove 'module.' prefix if model was trained with DataParallel
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state_dict)
        self.model.eval() 
        
        sz = CONFIG["IMAGE_SIZE"]
        interp = transforms.InterpolationMode.LANCZOS
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # Test Time Augmentation (TTA) Transforms
        self.base_transform = transforms.Compose([
            PadToSquare(), transforms.Resize((sz, sz), interpolation=interp), transforms.ToTensor(), norm
        ])
        self.flip_transform = transforms.Compose([
            PadToSquare(), transforms.Resize((sz, sz), interpolation=interp), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm
        ])

    def run_analysis(self, pil_img):
        try:
            with torch.no_grad():
                tensor_base = self.base_transform(pil_img).unsqueeze(0).to(self.device)
                tensor_flip = self.flip_transform(pil_img).unsqueeze(0).to(self.device)
                
                # BFloat16 Inference for extreme speed on RTX 50-series without underflow
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    logit_base = self.model(tensor_base).item()
                    logit_flip = self.model(tensor_flip).item()
                
            # Average logits from TTA
            avg_logit = (logit_base + logit_flip) / 2.0
            ai_real_score = torch.sigmoid(torch.tensor(avg_logit)).item()
            
            logger.info(f"🔎 [Forensic Analysis]: Logit = {avg_logit:.4f} -> Real Probability = {ai_real_score*100:.2f}%")
            return max(0.0001, min(0.9999, ai_real_score))
            
        except Exception as e:
            logger.error(f"❌ [Compute Error] Inference failed. Details:\n{traceback.format_exc()}")
            return None