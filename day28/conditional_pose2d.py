import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SEED = 4
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------- Data: noisy, partially observed 2D shapes ---------------------

def make_templates():
    xs = np.linspace(-0.9, 0.9, 12)
    bar = np.concatenate([np.stack([xs, np.full_like(xs, -0.12)], -1),
                          np.stack([xs, np.full_like(xs, +0.12)], -1)], 0)
    # L-shaped object (non-symmetric) -> has a uniquely observable heading
    lx = np.linspace(-1, 1, 12)
    ly = np.linspace(-0.6, 1.2, 12)
    angle = np.concatenate([np.stack([lx, np.full(12, -0.6)], -1),
                            np.stack([np.full(12, -1.0), ly], -1)])
    angle -= angle.mean(axis=0, keepdims=True)
    return bar.astype(np.float32), angle.astype(np.float32)

BAR, ANGLE = make_templates()

def make_data(n, seed):
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 20, 2), dtype=np.float32)
    labels = np.zeros((n, 4), dtype=np.float32)
    kinds = rng.integers(0, 2, size=n)
    poses = rng.uniform(-np.pi, np.pi, size=n)
    centers = rng.uniform(-0.75, 0.75, size=(n, 2))
    for i in range(n):
        template = BAR if kinds[i] == 0 else ANGLE
        idx = rng.choice(len(template), size=20, replace=False)
        local = template[idx] * rng.uniform(0.9, 1.1)
        c, s = np.cos(poses[i]), np.sin(poses[i])
        R = np.array([[c, -s], [s, c]])
        pts[i] = (local @ R.T + centers[i] + rng.normal(0, 0.025, size=(20,2)))
        rng.shuffle(pts[i])  # unordered point cloud
        labels[i] = [centers[i,0], centers[i,1], c, s]
    return (torch.from_numpy(pts), torch.from_numpy(labels),
            torch.from_numpy(kinds.astype(np.int64)))

# --------------------- Conditional 2D pose model ---------------------
class PoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_net = nn.Sequential(nn.Linear(2,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.encoder = nn.Sequential(nn.Linear(128+9,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU())
        self.reg = nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,4))
        self.denoise = nn.Sequential(nn.Linear(96+4+1,128),nn.SiLU(),
                                     nn.Linear(128,128),nn.SiLU(),nn.Linear(128,4))

    def encode(self, pts):
        h = self.point_net(pts)
        x, y = pts[:,:,0], pts[:,:,1]
        mx, my = x.mean(1), y.mean(1)
        dx, dy = x - mx[:,None], y - my[:,None]
        stats = torch.stack([mx, my, (dx*dx).mean(1), (dy*dy).mean(1),
                             (dx*dy).mean(1), (dx**3).mean(1),
                             (dx*dx*dy).mean(1), (dx*dy*dy).mean(1),
                             (dy**3).mean(1)], dim=1)
        return self.encoder(torch.cat([h.mean(1),h.max(1).values,stats],dim=1))

    def predict_noise(self, h, noisy_pose, t):
        tau = (t.float() / (N_STEPS-1))[:,None]
        return self.denoise(torch.cat([h, noisy_pose, tau], dim=1))

N_STEPS=64
betas = torch.linspace(0.0005, 0.20, N_STEPS, device=DEVICE)
alphas = 1-betas
alpha_bars = torch.cumprod(alphas,0)

def regression_loss(pred, target, kind):
    # Rotation ambiguity for bars deliberately NOT resolved by supervision.
    return F.mse_loss(pred,target)

def train():
    X, Y, kind = make_data(5000,111)
    X, Y, kind = X.to(DEVICE), Y.to(DEVICE), kind.to(DEVICE)
    net = PoseModel().to(DEVICE)
    optim = torch.optim.Adam(net.parameters(), lr=2e-3)
    batch=128

    print('DEVICE:',DEVICE, 'terminal alpha_bar:',float(alpha_bars[-1]))
    print('\nPRETRAIN regression:')
    for step in range(501):
        ix = torch.randint(len(X),(batch,),device=DEVICE)
        h = net.encode(X[ix])
        reg = net.reg(h)
        loss = regression_loss(reg,Y[ix],kind[ix])
        optim.zero_grad(); loss.backward(); optim.step()
        if step%250==0: print(f'  step={step:4d} reg_loss={loss.item():.4f}')

    print('\nJOINT training:')
    for step in range(1201):
        ix = torch.randint(len(X),(batch,),device=DEVICE)
        h = net.encode(X[ix]); y = Y[ix]
        t = torch.randint(N_STEPS,(batch,),device=DEVICE)
        eps = torch.randn_like(y)
        a = alpha_bars[t,None]
        noisy = a.sqrt()*y + (1-a).sqrt()*eps
        eps_hat = net.predict_noise(h,noisy,t)
        loss_diff = F.mse_loss(eps_hat,eps)
        loss_reg = regression_loss(net.reg(h),y,kind[ix])
        loss = loss_diff + 0.5*loss_reg
        optim.zero_grad(); loss.backward(); optim.step()
        if step%400==0: print(f'  step={step:4d} diff={loss_diff.item():.4f} reg={loss_reg.item():.4f}')
    return net

@torch.no_grad()
def ddpm_sample(net, cloud, num_hyp=1):
    net.eval()
    B = cloud.shape[0]
    h = net.encode(cloud).repeat_interleave(num_hyp,dim=0)
    x = torch.randn((B*num_hyp,4), device=DEVICE)
    for i in reversed(range(N_STEPS)):
        ti = torch.full((len(x),),i,device=DEVICE,dtype=torch.long)
        eps = net.predict_noise(h,x,ti)
        mean = (x - betas[i]/torch.sqrt(1-alpha_bars[i])*eps)/torch.sqrt(alphas[i])
        if i:
            var = betas[i]*(1-alpha_bars[i-1])/(1-alpha_bars[i])
            x = mean + var.sqrt()*torch.randn_like(x)
        else:
            x = mean
    return x.view(B,num_hyp,4)

@torch.no_grad()
def print_metrics(title, pred, truth, kinds):
    # pred: (B,4), not an oracle best-of-many
    xy_err = torch.linalg.vector_norm(pred[:,:2]-truth[:,:2],dim=1)
    pdir = F.normalize(pred[:,2:4],dim=1,eps=1e-8)
    rdir = truth[:,2:4]
    dot = (pdir*rdir).sum(1).clamp(-1,1)
    # arrow: theta mod 2pi, symmetric bar: theta mod pi
    cos_sim = torch.where(kinds==0,dot.abs(),dot)
    deg_err = torch.rad2deg(torch.acos(cos_sim.clamp(-1,1)))
    print(f'{title:<14s} center={xy_err.mean():.3f} angle={deg_err.mean():5.1f}° '
          f'bar={deg_err[kinds==0].mean():5.1f}° L={deg_err[kinds==1].mean():5.1f}° '
          f'<15°={(deg_err<15).float().mean()*100:5.1f}%')
    return xy_err,deg_err

def main():
    net = train()
    X,Y,kinds = make_data(800,222)  # Entirely unseen scenes
    X,Y,kinds = X.to(DEVICE),Y.to(DEVICE),kinds.to(DEVICE)
    net.eval()
    with torch.no_grad():
        reg = net.reg(net.encode(X))
    diff = ddpm_sample(net,X,num_hyp=1)[:,0,:]
    print('\n=== UNSEEN TEST (one prediction per input) ===')
    print_metrics('Regression',reg,Y,kinds)
    print_metrics('Diffusion',diff,Y,kinds)
    print('\nRotation-vector norm on symmetric bars:')
    print(' regression:',float(reg[kinds==0,2:4].norm(dim=1).mean()),
          'diffusion:',float(diff[kinds==0,2:4].norm(dim=1).mean()))
    # Fix one identical observed shape and repeat sampling 128 times
    bar_idx = int(torch.where(kinds==0)[0][0])
    fixed_cloud = X[bar_idx:bar_idx+1]
    variants = ddpm_sample(net,fixed_cloud,num_hyp=128)[0]
    dirs = F.normalize(variants[:,2:4],dim=1)
    gt = Y[bar_idx,2:4]
    signed = (dirs*gt).sum(1)
    eq_err = torch.rad2deg(torch.acos(signed.abs().clamp(-1,1)))
    print('\n=== SAME SYMMETRIC BAR, 128 pose hypotheses ===')
    print('  aligned orientation :',int((signed>=0).sum()),
          'flipped 180-degree:',int((signed<0).sum()))
    print('  mean symmetry-aware error:',float(eq_err.mean()))
    # Visual comparison for one symmetric and one asymmetric shape
    try:
        import matplotlib.pyplot as plt
        l_idx = int(torch.where(kinds==1)[0][0])
        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        for row, idx in enumerate([bar_idx, l_idx]):
            for col, (poses, title) in enumerate([(Y,'Ground truth'),(reg,'Regression'),(diff,'Diffusion')]):
                ax = axes[row,col]
                pts = X[idx].detach().cpu().numpy()
                pose = poses[idx].detach().cpu().numpy()
                ax.scatter(pts[:,0],pts[:,1],s=15,alpha=.7)
                direction = pose[2:4] / (np.linalg.norm(pose[2:4])+1e-8)
                ax.arrow(pose[0],pose[1],0.65*direction[0],0.65*direction[1],
                         head_width=.09, length_includes_head=True, color='tab:orange')
                ax.set(xlim=(-2.2,2.2),ylim=(-2.2,2.2),aspect='equal',
                       title=f'{"Bar (symmetric)" if row==0 else "L (asymmetric)"}: {title}')
                ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig('pose2d_examples.png',dpi=160)
        plt.close(fig)
        print('Saved: pose2d_examples.png')
    except ImportError:
        pass
    torch.save(net.state_dict(),'pose2d_model.pt')
    print('\nSaved: pose2d_model.pt')

if __name__ == '__main__': main()
