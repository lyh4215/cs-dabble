"""Day31: Image-only vs Raw Motion Vectors vs Prewarped Motion Vectors.
No downloads. Requires torch and numpy.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 31
torch.manual_seed(SEED)
torch.set_num_threads(min(torch.get_num_threads(), 4))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
H, W = 32, 48
Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
X, Y = X.float()[None, None], Y.float()[None, None]


def scenes(n, seed, split):
    g = torch.Generator().manual_seed(seed)
    def uniform(lo, hi, shape):
        return lo + (hi-lo) * torch.rand(shape, generator=g)
    dx = torch.randint(4, 14, (n,1,1,1), generator=g).float() if split == 'train' else torch.randint(12, 19, (n,1,1,1), generator=g).float()
    left = torch.randint(4, 13, (n,1,1,1), generator=g).float()
    cy = torch.randint(12, 22, (n,1,1,1), generator=g).float()
    phase = uniform(0, 6.28, (n,1,1,1))
    bg = torch.cat([
        .25 + .12*torch.sin(.22*X + .15*Y + phase),
        .35 + .10*torch.cos(.17*X - .13*Y + phase),
        .40 + .10*torch.sin(.11*Y + .19*X - phase),
    ], dim=1)
    color = uniform(.3, .85, (n,3,1,1))
    lightx = uniform(8, 37, (n,1,1,1))
    lightdx = uniform(-9, 9, (n,1,1,1))
    def render(t):
        mask = ((X >= left+dx*t)&(X < left+dx*t+11)&(Y >= cy)&(Y < cy+13))
        image = torch.where(mask.expand(-1,3,-1,-1), color.expand(-1,-1,H,W), bg)
        glow = .25*torch.exp(-((X-lightx-lightdx*t)**2+(Y-7)**2)/(2*4.0**2))
        image += torch.cat([glow,glow*.25,glow*.05],dim=1)
        return image.clamp(0,1),mask.float()
    i0,m0=render(0); i1,m1=render(1); mid,_=render(.5)
    images=torch.cat([i0,i1],dim=1)
    # Per-pixel source and target motion vectors: 0 for static background.
    mv0=(dx/W*m0).expand(-1,1,H,W)
    mv1=(-dx/W*m1).expand(-1,1,H,W)
    raw=torch.cat([mv0,mv1,m0,m1],dim=1)
    # Warping is a known geometric operator; not learned by the CNN.
    w0=[];w1=[];wm0=[];wm1=[]
    for b in range(n):
        half=int(dx[b,0,0,0].item())//2
        w0.append(torch.roll(i0[b]*m0[b], half, dims=-1))
        w1.append(torch.roll(i1[b]*m1[b], -half, dims=-1))
        wm0.append(torch.roll(m0[b], half, dims=-1))
        wm1.append(torch.roll(m1[b], -half, dims=-1))
    warped=torch.cat([torch.stack(w0),torch.stack(w1),torch.stack(wm0),torch.stack(wm1)],dim=1)
    return images,raw,warped,mid


class CNN(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(channels,48,3,padding=1),nn.ReLU(),
            nn.Conv2d(48,48,3,padding=1),nn.ReLU(),
            nn.Conv2d(48,48,3,padding=1),nn.ReLU(),
            nn.Conv2d(48,3,3,padding=1))
    def forward(self,x):
        baseline=.5*(x[:,:3]+x[:,3:6])
        return (baseline+self.net(x)).clamp(0,1)


def assemble(images,raw,warped,kind):
    if kind=='image': return images
    if kind=='raw': return torch.cat([images,raw],dim=1)
    if kind=='warp': return torch.cat([images,raw,warped],dim=1)
    raise ValueError(kind)


def psnr(pred,target):
    # Mean pixel MSE over all scenes, then PSNR.
    mse=((pred-target)**2).mean().item()
    return -10*np.log10(max(mse,1e-12))


def train(kind,steps=100):
    torch.manual_seed(SEED)
    channels={'image':6,'raw':10,'warp':18}[kind]
    net=CNN(channels).to(DEVICE)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    print(f'\nTraining {kind}, params={sum(p.numel() for p in net.parameters())}')
    for step in range(steps+1):
        imgs,raw,warp,target=scenes(12,1000+step,'train')
        x=assemble(imgs,raw,warp,kind).to(DEVICE)
        target=target.to(DEVICE)
        prediction=net(x)
        loss=F.l1_loss(prediction,target)
        opt.zero_grad();loss.backward();opt.step()
        if step%25==0:
            print(f'{kind:6s} step={step:3d} L1={loss.item():.4f}')
    return net.eval()


def main():
    print('Device:', DEVICE)
    models={kind:train(kind) for kind in ('image','raw','warp')}
    print('\n=== HELD-OUT PSNR (dB) ===')
    print(f'{"split":7s} {"blend":>8s} {"image":>8s} {"raw MV":>8s} {"MV+warp":>8s}')
    for split in ('train','test'):
        imgs,raw,warp,target=scenes(32,98765,split)
        target=target.to(DEVICE)
        results={'blend':psnr(.5*(imgs[:,:3]+imgs[:,3:6]),target.cpu())}
        with torch.no_grad():
            for kind,net in models.items():
                x=assemble(imgs,raw,warp,kind).to(DEVICE)
                results[kind]=psnr(net(x),target)
        print(f'{split:7s} {results["blend"]:8.2f} {results["image"]:8.2f} {results["raw"]:8.2f} {results["warp"]:8.2f}')
    print('\nNote: integer half-pixel warping approximation (dx//2); raw and warped models use more inputs/parameters.')

if __name__=='__main__': main()
