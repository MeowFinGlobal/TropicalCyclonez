import os, random, torch, numpy as np
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size, batch_size, epochs, lr = 512, 8, 20, 1e-3
sample_dir = "fusion_samples"
os.makedirs(sample_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()
])

def get_edge_strength(tensor):
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=tensor.device).reshape(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=tensor.device).reshape(1,1,3,3)
    gx = F.conv2d(tensor.unsqueeze(0), sobel_x, padding=1)
    gy = F.conv2d(tensor.unsqueeze(0), sobel_y, padding=1)
    edge = torch.sqrt(gx ** 2 + gy ** 2).squeeze(0)  # (1, H, W)
    return edge

class FusionDataset(Dataset):
    def __init__(self, cyclone_dir, tc_space_dir, fusion_dir=None, transform=None):
        self.cyclone = sorted([f for f in os.listdir(cyclone_dir) if f.lower().endswith(('.png', '.jpg'))])
        self.tc = sorted([f for f in os.listdir(tc_space_dir) if f.lower().endswith(('.png', '.jpg'))])
        self.fusion = sorted(os.listdir(fusion_dir)) if fusion_dir else None
        self.cyclone_dir, self.tc_dir, self.fusion_dir, self.transform = cyclone_dir, tc_space_dir, fusion_dir, transform

    def __getitem__(self, idx):
        def load(folder, file): return self.transform(Image.open(os.path.join(folder, file)).convert("L"))
        cyclone = load(self.cyclone_dir, self.cyclone[idx])
        refs = torch.stack([load(self.tc_dir, self.tc[i]) for i in random.sample(range(len(self.tc)), 3)]).mean(0)
        if self.fusion_dir:
            fusion = load(self.fusion_dir, self.fusion[idx])
        else:
            edge_c = get_edge_strength(cyclone)  # (1,H,W)
            edge_r = get_edge_strength(refs)
            # 归一化
            edge_c = (edge_c - edge_c.min()) / (edge_c.max() - edge_c.min() + 1e-8)
            edge_r = (edge_r - edge_r.min()) / (edge_r.max() - edge_r.min() + 1e-8)
            # 平滑权重，sigmoid增强边缘敏感性
            alpha = 3.0  # 控制边缘引导的强弱，越大越极端，建议2~5之间试
            edge_weight = torch.sigmoid(alpha * (edge_r - edge_c))  # [0,1]
            fusion = edge_weight * refs + (1 - edge_weight) * cyclone
        return cyclone, refs, fusion

    def __len__(self): return min(len(self.cyclone), len(self.tc))

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, 1, 1), nn.ReLU(inplace=True))
        self.enc = nn.ModuleList([block(2, 64), block(64,128), block(128,256), block(256,512)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(512, 1024)
        self.up = nn.ModuleList([nn.ConvTranspose2d(1024, 512, 2,2), nn.ConvTranspose2d(512,256,2,2), nn.ConvTranspose2d(256,128,2,2), nn.ConvTranspose2d(128,64,2,2)])
        self.dec = nn.ModuleList([block(1024,512), block(512,256), block(256,128), block(128,64)])
        self.out_conv = nn.Conv2d(64, 1, 1)
    def forward(self, x1, x2):
        x, encs = torch.cat([x1, x2], 1), []
        for e in self.enc: x = e(x); encs.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(4): x = self.up[i](x); x = torch.cat([x, encs[3-i]], 1); x = self.dec[i](x)
        return self.out_conv(x)

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps = eps
    def forward(self, x, y): return torch.mean(torch.sqrt((x - y) ** 2 + self.eps))

class SSIMLoss(nn.Module):
    def __init__(self, w=11):
        super().__init__()
        self.window = self.create_window(w).to(device)
        self.window_size = w; self.channel = 1
    def gaussian_window(self, w, sigma=1.5): g = torch.Tensor([np.exp(-(x-w//2)**2/(2*sigma**2)) for x in range(w)]); return g/g.sum()
    def create_window(self, s, c=1): a = self.gaussian_window(s).unsqueeze(1); b = a@a.t(); return b.expand(c,1,s,s).contiguous()
    def forward(self, x, y):
        mu1 = F.conv2d(x, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(y, self.window, padding=self.window_size//2, groups=self.channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
        sigma1_sq = F.conv2d(x*x, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(y*y, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(x*y, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
        return 1 - ssim_map.mean()

dataset = FusionDataset("test", "2", fusion_dir=None, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = UNet().to(device)
criterion, ssim_loss = CharbonnierLoss(), SSIMLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for i, (cyc, tc, target) in enumerate(dataloader):
        cyc, tc, target = cyc.to(device), tc.to(device), target.to(device)
        out = model(cyc, tc)
        loss = criterion(out, target) + 0.5 * ssim_loss(out, target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    if epoch >= 4:
        with torch.no_grad():
            sample = model(cyc[:1], tc[:1])
            sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-5) * (255 + 95) - 95
            utils.save_image(sample/255., f"{sample_dir}/epoch_{epoch+1}.jpg") 

#Enceladus
