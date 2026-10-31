import argparse
import time
import torch
import torch.nn.functional as F


def attention(x: torch.Tensor, use_sdpa=True) -> torch.Tensor:
    # Toy self-attention with Q=K=V=x; no trainable parameters.
    d = x.shape[-1]
    if use_sdpa:
        return F.scaled_dot_product_attention(x, x, x, dropout_p=0.0)
    weights = torch.softmax(x @ x.transpose(-1, -2) / (d ** 0.5), dim=-1)
    return weights @ x


def timed(fn, device, runs):
    for _ in range(5):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=1024)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--keep', type=float, default=0.5)
    ap.add_argument('--runs', type=int, default=30)
    args = ap.parse_args()
    assert 0 < args.keep <= 1 and args.tokens >= 2
    torch.manual_seed(42)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, args.tokens, args.dim, device=dev)
    # Artificially amplify some tokens, only to make scores nonuniform.
    x[:, :args.tokens // 8] *= 2.0
    with torch.inference_mode():
        y1 = x + attention(x)
        # Shift across the first residual attention block; educational proxy.
        scores = (y1 - x).norm(dim=-1)[0]
        keep_n = max(1, round(args.tokens * args.keep))
        chosen = scores.topk(keep_n).indices.sort().values
        random_chosen = torch.randperm(args.tokens, device=dev)[:keep_n].sort().values

        full_next = y1 + attention(y1)
        pruned = y1[:, chosen]
        random_pruned = y1[:, random_chosen]
        pruned_next = pruned + attention(pruned)
        random_next = random_pruned + attention(random_pruned)
        
        # Changes to outputs of REMAINING tokens, not task accuracy.
        err_shift = (pruned_next - full_next[:, chosen]).norm() / full_next[:, chosen].norm().clamp_min(1e-9)
        err_random = (random_next - full_next[:, random_chosen]).norm() / full_next[:, random_chosen].norm().clamp_min(1e-9)
        t_full = timed(lambda: attention(y1), dev, args.runs)
        t_pruned = timed(lambda: attention(pruned), dev, args.runs)
        t_score = timed(lambda: (y1 - x).norm(dim=-1), dev, args.runs)
        t_map = timed(lambda: attention(y1, use_sdpa=False), dev, args.runs)
    print('device:', dev, 'tokens:', args.tokens, 'dim:', args.dim, 'kept:', keep_n)
    print('shift scores: min=%.4f median=%.4f max=%.4f' % (scores.min(), scores.median(), scores.max()))
    print('Relative output error on retained tokens (lower is better; not accuracy):')
    print('  shift  %.6f' % err_shift.item())
    print('  random %.6f' % err_random.item())
    print('Latency, attention only, ms:')
    print('  SDPA full      %.4f' % t_full)
    print('  SDPA pruned    %.4f' % t_pruned)
    print('  scoring only   %.4f' % t_score)
    print('  explicit map   %.4f' % t_map)
    print('NOTE: SDPA backend depends on device/PyTorch. This is not official paper reproduction.')


if __name__ == '__main__':
    main()
