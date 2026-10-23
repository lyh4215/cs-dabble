"""BigDyL Deep Dive 05 — Counterfactual expert competence vs routing.
Run next to moe_shift.py and lda_vs_pca.py. Python 3.11, torch/numpy.
For each test sample evaluate EVERY trained expert, not just routed expert.
Oracle expert selection uses test labels, for diagnosis only (not deployable).
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from moe_shift import DOMAINS, MoE, Router, data, pca_basis, set_seed, train
from lda_vs_pca import ProjectedRouter, lda_basis, shifted_dataset


@torch.no_grad()
def counterfactual(model, dataset):
    x, y, domains = dataset
    model.eval()
    routed_logits, probs, routed = model(x)
    all_logits = torch.stack([expert(x) for expert in model.experts], dim=1)  # [N, E, 2]
    all_preds = all_logits.argmax(-1)  # [N, E]
    correct = all_preds.eq(y[:, None])
    routed_acc = (routed_logits.argmax(-1) == y).float().mean().item()
    forced_acc = correct.float().mean(0).tolist()
    # Classification oracle: at least one expert can classify correctly.
    oracle_acc = correct.any(dim=1).float().mean().item()
    # Oracle advantage includes errors of the router and failures of chosen experts;
    # not all of it is necessarily attributable to routing alone.
    routed_correct = routed_logits.argmax(-1).eq(y)
    oracle_gap = oracle_acc - routed_acc
    matrix = [[float('nan')]*DOMAINS for _ in range(DOMAINS)]
    routed_hist = [[0.0]*DOMAINS for _ in range(DOMAINS)]
    domain_routed_acc = [float('nan')]*DOMAINS
    for d in range(DOMAINS):
        mask = domains == d
        if not mask.any():
            continue
        matrix[d] = correct[mask].float().mean(0).tolist()
        routed_hist[d] = (torch.bincount(routed[mask], minlength=DOMAINS).float() / mask.sum()).tolist()
        domain_routed_acc[d] = routed_correct[mask].float().mean().item()
    # Does routing choose an expert that's best for the domain on this split?
    optimal_by_domain = [int(np.nanargmax(matrix[d])) for d in range(DOMAINS)]
    alignment = sum((routed[domains == d] == optimal_by_domain[d]).sum().item() for d in range(DOMAINS)) / len(x)
    return dict(routed_acc=routed_acc, oracle_acc=oracle_acc, oracle_gap=oracle_gap,
                forced_acc=forced_acc, matrix=matrix, routing=routed_hist,
                domain_routed_acc=domain_routed_acc, domain_best=optimal_by_domain,
                domain_best_alignment=alignment)


def build_model(name, train_set, seed, ridge=0.05, hidden=32):
    if name == 'linear':
        router = Router('linear')
    else:
        if name == 'pca2':
            basis, center = pca_basis(train_set[0], 2)
        elif name == 'label_lda1':
            basis, center = lda_basis(train_set[0], train_set[1], 1, ridge)
        elif name == 'domain_lda2':
            basis, center = lda_basis(train_set[0], train_set[2], 2, ridge)
        else:
            raise ValueError(name)
        router = ProjectedRouter(basis, center, hidden=hidden)
    return MoE(router)


def run(args):
    records=[]
    names=('linear','pca2','label_lda1','domain_lda2')
    for seed in range(args.seed, args.seed+args.seeds):
        set_seed(seed)
        training=data(args.train_n, seed)
        base=data(args.test_n, 10000+seed)
        datasets={'iid':base, 'domain_shift':shifted_dataset(base,'domain',args.shift)}
        for name in names:
            torch.manual_seed(seed)
            model=build_model(name, training, seed, ridge=args.ridge, hidden=args.hidden)
            train(model,training,args.epochs,args.batch,args.lr,args.balance)
            for split, dataset in datasets.items():
                r=counterfactual(model,dataset)
                print(f'\nseed={seed} {name} {split} routed={r["routed_acc"]:.3f} oracle={r["oracle_acc"]:.3f} gap={r["oracle_gap"]:.3f} align={r["domain_best_alignment"]:.3f}')
                print('Forced expert accuracy, rows=domain, columns=expert:')
                for d in range(DOMAINS):
                    print(f' D{d}: '+ ' '.join(f'{v:.3f}' for v in r['matrix'][d])+f' | router: '+ ' '.join(f'{v:.2f}' for v in r['routing'][d]))
                row={'seed':seed,'router':name,'split':split,'routed_acc':r['routed_acc'],
                     'oracle_acc':r['oracle_acc'],'oracle_gap':r['oracle_gap'],
                     'best_domain_alignment':r['domain_best_alignment']}
                for d in range(DOMAINS):
                    row[f'domain{d}_routed_acc']=r['domain_routed_acc'][d]
                    row[f'domain{d}_best_expert']=r['domain_best'][d]
                    for e in range(DOMAINS):
                        row[f'd{d}_e{e}_acc']=r['matrix'][d][e]
                        row[f'd{d}_e{e}_routefreq']=r['routing'][d][e]
                records.append(row)
    with open(args.out, 'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    print('\n=== AGGREGATE across seeds ===')
    for name in names:
        for split in datasets:
            selected=[r for r in records if r['router']==name and r['split']==split]
            print(f'{name:12} {split:12} '+' '.join(f'{metric}={np.mean([r[metric] for r in selected]):.3f}' for metric in ('routed_acc','oracle_acc','oracle_gap','best_domain_alignment')))
    print('Saved',args.out)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--train-n',type=int,default=2400)
    p.add_argument('--test-n',type=int,default=1200)
    p.add_argument('--shift',type=float,default=0.7)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--lr',type=float,default=0.002)
    p.add_argument('--balance',type=float,default=0.1)
    p.add_argument('--ridge',type=float,default=0.05)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--out',type=Path,default=Path('counterfactual_experts_results.csv'))
    run(p.parse_args())
