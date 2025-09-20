#    ∧__∧
#   ( ･ω･｡)つ━☆・*。
#   ⊂    ノ    ・゜+.
#   しーＪ   ★ AC GET ★
import torch
import torch.nn as nn
import numpy as np


torch.manual_seed(13)



N=nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))


x=torch.rand((1,10)) 
x.requires_grad_()

t_target=1

eps_real=0.8       
eps_grid=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5] 
# fgsm_step_eps=None  


def get_logits_pred(model, inp):
    with torch.no_grad():
        logits=model(inp)
        pred=logits.argmax(dim=1).item()
    return logits.detach().cpu().numpy(), pred


x_base=x.clone().detach()
x_base=x_base.requires_grad_(True)

# print("Original logits and pred (N(x)):")
# orig_logits, orig_pred=get_logits_pred(N, x_base)
# print(orig_logits, "pred =", orig_pred)


for eps in eps_grid:
    x_try=x_base.clone().detach().requires_grad_(True)
    loss_fn=nn.CrossEntropyLoss()
    out=N(x_try)
    loss=loss_fn(out, torch.tensor([t_target], dtype=torch.long))
    N.zero_grad()
    if x_try.grad is not None:
        x_try.grad.zero_()
    loss.backward()
    grad=x_try.grad.data
    adv=x_try.detach() - (eps - 1e-9) * grad.sign()
    adv=torch.max(torch.min(adv, x_base + eps_real), x_base - eps_real)
    adv=torch.clamp(adv, 0.0, 1.0).detach()
    # print("debug:",adv)
    logits_adv, pred_adv=get_logits_pred(N, adv)
    linf=torch.norm((x_base.detach() - adv), p=float('inf')).item()
    success=(pred_adv == t_target)
    print(f"[FGSM] eps={eps:.4f} -> pred={pred_adv}, success={success}, ||x'-x||_inf={linf:.6f}")
    print("  logits_adv =", logits_adv)
    if success:
        adv_x=adv
        break

# print("next task")

def targeted_pgd_attack(x_2, model, target, eps_2, alpha=0.02, iters=100, random_start=False):
    if random_start:
        adv=x_2 + (2*torch.rand_like(x_2) - 1.0) * eps_2
        adv=torch.clamp(adv, 0.0, 1.0).detach()
    else:
        adv=x_2.clone().detach()
    adv=adv.requires_grad_(True)

    for i in range(iters):
        out=model(adv)
        loss=nn.CrossEntropyLoss()(out, torch.tensor([target], dtype=torch.long))
        model.zero_grad()
        if adv.grad is not None:
            adv.grad.zero_()
        loss.backward()

        adv_data=adv.detach() - alpha * adv.grad.data.sign()
        # print("debug:",adv)
        # adv_data=adv-alpha*adv.grad.data.sign()
        adv_data=torch.max(torch.min(adv_data, x_2 + eps_2), x_2 - eps_2)
        adv_data=torch.clamp(adv_data, 0.0, 1.0)
        adv=adv_data.detach().requires_grad_(True)
        pred=model(adv).argmax(dim=1).item()

        # pred=model(adv).argmax(dim=1)
        if pred == target:
            return adv.detach(), True, i+1
    return adv.detach(), False, iters


pgd_configs=[
    # {"alpha": 0.01, "iters": 100, "random_start": False},
    {"alpha": 0.01, "iters": 200, "random_start": False},
    {"alpha": 0.01, "iters": 400, "random_start": True},
    # {"alpha": 0.02, "iters": 100, "random_start": False},
    {"alpha": 0.02, "iters": 200, "random_start": True},
    {"alpha": 0.2, "iters": 200, "random_start": True},
]
pgd_success=False
for cfg in pgd_configs:
    adv_pgd, succeeded, used_iters=targeted_pgd_attack(x_base.detach(), N, t_target,
                                                        eps_real, alpha=cfg["alpha"], iters=cfg["iters"],
                                                        random_start=cfg["random_start"])
    
    logits_pgd, pred_pgd=get_logits_pred(N, adv_pgd)
    # logits_pdg=get_logits_pred(N, adv_pgd)
    linf=torch.norm((x_base.detach() - adv_pgd), p=float('inf')).item()
    print(f"[PGD] alpha={cfg['alpha']}, iters={cfg['iters']}, random_start={cfg['random_start']} -> pred={pred_pgd}, succ={succeeded}, iters_used={used_iters}, ||.||_inf={linf:.6f}")
    print("  logits_pgd =", logits_pgd)
    if succeeded:
        pgd_success=True
        adv_x=adv_pgd
        break
