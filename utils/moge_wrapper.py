import torch.nn as nn
import torch
import cv2
import torch.nn.functional as F
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[1]
MOGE_PATH = PROJECT_ROOT / "third_repos" / "moge"
if str(MOGE_PATH) not in sys.path:
    sys.path.insert(0, str(MOGE_PATH))

from moge.model.v2 import MoGeModel

CHECKPOINTS_DIR = PROJECT_ROOT / "third_repos" / "moge" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINTS_DIR / "moge-2-vits-normal.pt"

class MoGeWrapper(nn.Module):
    def __init__(self, device, process_size=(608,800)):
        super().__init__()
        self.net = MoGeModel.from_pretrained(CHECKPOINT_PATH).to(device).eval()
        self.device = device
        self.process_size = process_size

    @torch.inference_mode()
    def extract(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img = img.to(self.device)
        output = self.net.infer(img, apply_mask=False)

        normal = output["normal"]
        depth = output["depth"]

        depth = F.interpolate(depth[None, None], self.process_size, mode="bilinear", align_corners=True)[0, 0]
        normal = F.interpolate(normal.permute(2, 0, 1)[None], self.process_size, mode="bilinear", align_corners=True)[0]
        normal = F.normalize(normal, dim=0).permute(1, 2, 0)
        depth = (depth - depth.min()) / (depth.max() - depth.min()).clamp_min(1e-6) * 255.0

        return depth, normal
