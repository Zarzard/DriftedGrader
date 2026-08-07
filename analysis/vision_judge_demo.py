"""Vision reward-model / judge under a drifting standard (frozen ImageNet ResNet18 & DINOv2).

Mirrors the LLM-judge demo in the image modality. CIFAR images x have interpretable attributes
a(x) in R^6 (brightness, contrast, colorfulness, saturation, warm-cool, edge-density). The
standard w_t drifts (vivid/warm -> dark/edgy/cool). A frozen vision encoder e(x) is the reward
representation; a Bayesian linear head gives scalar reward + VaR for selection. We ask, per
encoder:
  (1) full-embedding held-out ridge probe: is the DRIFT direction covered ~ as well as random?
  (2) selection regret per drift stage (does it rise with drift?)
  (3) narrow-head sweep: regret vs head dim k (how narrow to fail, in vision).

Run: HF_HOME=... TORCH_HOME=... python -m analysis.vision_judge_demo --out <dir> [--n 2000]
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
from drifted_grader.metrics import rank_correlation
from analysis.llm_judge_demo import _std, _pca, _fit_lb, make_projection
from analysis.llm_judge_refine import ridge_r2

ATTRS = ["brightness", "contrast", "colorfulness", "saturation", "warm_cool", "edge_density"]
CIFAR = "/scratch/general/vast/u1520755/cifar"


def load_images(n):
    import torchvision
    ds = torchvision.datasets.CIFAR10(CIFAR, train=False, download=False)
    return [np.array(ds[i][0]) for i in range(min(n, len(ds)))]      # list of HWC uint8


def img_attrs(im):
    x = im.astype(float) / 255.0; R, G, B = x[..., 0], x[..., 1], x[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    rg = R - G; yb = 0.5 * (R + G) - B
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    mx = x.max(-1); mn = x.min(-1); sat = ((mx - mn) / (mx + 1e-6)).mean()
    edge = (np.abs(np.diff(lum, axis=1)).mean() + np.abs(np.diff(lum, axis=0)).mean()) / 2
    return np.array([lum.mean(), lum.std(), colorful, sat, R.mean() - B.mean(), edge])


def encode_resnet(imgs):
    import torch, torchvision.transforms as T, torchvision.models as M
    m = M.resnet18(weights="IMAGENET1K_V1"); m.fc = torch.nn.Identity(); m.eval().cuda()
    tf = T.Compose([T.ToPILImage(), T.Resize(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), 128):
            b = torch.stack([tf(im) for im in imgs[i:i + 128]]).cuda()
            out.append(m(b).cpu().numpy())
    return np.concatenate(out, 0)


def encode_dino(imgs):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    m = AutoModel.from_pretrained("facebook/dinov2-base", torch_dtype=torch.float16).eval().cuda()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), 128):
            pil = [Image.fromarray(im) for im in imgs[i:i + 128]]
            enc = proc(images=pil, return_tensors="pt").to("cuda")
            enc = {k: v.half() for k, v in enc.items()}
            out.append(m(**enc).last_hidden_state[:, 0].float().cpu().numpy())     # CLS
    return np.concatenate(out, 0)


def analyze_encoder(name, E, A, w0, wE, rng, stages=9, npairs=3000):
    u = wE - w0; u /= np.linalg.norm(u); d = A.shape[1]
    cov = {"drift_u": ridge_r2(E, A @ u)}
    cov["random(mean)"] = float(np.mean([ridge_r2(E, A @ (lambda r: r / np.linalg.norm(r))(rng.standard_normal(d)), seed=i) for i in range(15)]))
    cov["random(sd)"] = float(np.std([ridge_r2(E, A @ (lambda r: r / np.linalg.norm(r))(rng.standard_normal(d)), seed=100 + i) for i in range(15)]))
    Epca = _std(_pca(E, min(64, E.shape[1])))
    scaleE = 1.0 / (np.sqrt((np.diff(Epca[rng.integers(0, len(Epca), (5000, 2))], 1)[:, 0] ** 2).sum(1).mean()) + 1e-8)
    per_stage = []
    for s in range(stages):
        frac = s / (stages - 1); wt = (1 - frac) * w0 + frac * wE; wt /= np.linalg.norm(wt)
        vt = A @ wt
        ij = rng.integers(0, len(A), (npairs, 2)); ij = ij[ij[:, 0] != ij[:, 1]]
        y = (rng.random(len(ij)) < 1.0 / (1.0 + np.exp(-10.0 * ((A[ij[:, 0]] - A[ij[:, 1]]) @ wt)))).astype(float)
        dE = Epca[ij[:, 0]] - Epca[ij[:, 1]]
        lb = _fit_lb(dE, y, Epca, scaleE, np.random.default_rng(s + 1))
        hi, lo = vt.max(), vt.min()
        per_stage.append(float((hi - vt[int(np.argmax(lb))]) / (hi - lo + 1e-9)))
    # narrow-head sweep (regret averaged over stages) at fixed mid drift
    ksweep = {}
    for k in (2, 4, 8, 16, 32, 64):
        if k > E.shape[1]:
            continue
        Pk = make_projection(Epca.shape[1], min(k, Epca.shape[1]), rng)
        regs = []
        for s in range(stages):
            frac = s / (stages - 1); wt = (1 - frac) * w0 + frac * wE; wt /= np.linalg.norm(wt); vt = A @ wt
            ij = rng.integers(0, len(A), (npairs, 2)); ij = ij[ij[:, 0] != ij[:, 1]]
            y = (rng.random(len(ij)) < 1.0 / (1.0 + np.exp(-10.0 * ((A[ij[:, 0]] - A[ij[:, 1]]) @ wt)))).astype(float)
            dE = Epca[ij[:, 0]] - Epca[ij[:, 1]]
            lb = _fit_lb(dE, y, Epca, scaleE, np.random.default_rng(s + 1), P=Pk)
            hi, lo = vt.max(), vt.min(); regs.append((hi - vt[int(np.argmax(lb))]) / (hi - lo + 1e-9))
        ksweep[k] = float(np.mean(regs))
    slope = float(np.polyfit(np.arange(stages), per_stage, 1)[0])
    return dict(name=name, cov=cov, regret_early=float(np.mean(per_stage[:stages // 3])),
                regret_late=float(np.mean(per_stage[2 * stages // 3:])), slope=slope, ksweep=ksweep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); rng = np.random.default_rng(0)
    imgs = load_images(args.n)
    A = _std(np.stack([img_attrs(im) for im in imgs]))
    print(f"images {len(imgs)}  attrs {A.shape}", flush=True)
    w0 = np.array([+1.0, +0.5, +1.0, +1.0, +0.5, 0.0]); w0 /= np.linalg.norm(w0)   # vivid/warm
    wE = np.array([-1.0, +1.0, -0.5, -0.5, -1.0, +1.0]); wE /= np.linalg.norm(wE)  # dark/edgy/cool
    res = []
    # oracle on true attributes (reference)
    res.append(analyze_encoder("oracle_attr", A.copy(), A, w0, wE, np.random.default_rng(1)))
    for enc, fn in [("resnet18_imagenet", encode_resnet), ("dinov2_base", encode_dino)]:
        E = _std(fn(imgs)); print(f"{enc} embeddings {E.shape}", flush=True)
        res.append(analyze_encoder(enc, E, A, w0, wE, np.random.default_rng(2)))
    json.dump(res, open(os.path.join(args.out, "vision.json"), "w"), indent=2)
    L = ["# Vision reward-model under a drifting standard (frozen ImageNet encoders)\n",
         f"CIFAR n={len(imgs)}  attrs={ATTRS}  drift: vivid/warm -> dark/edgy/cool\n",
         "| encoder | cov_r2(drift_u) | cov_r2(random) | regret early | regret late | slope/stage |",
         "|---|---|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['name']} | {r['cov']['drift_u']:.3f} | {r['cov']['random(mean)']:.3f}±{r['cov']['random(sd)']:.3f} "
                 f"| {r['regret_early']:.3f} | {r['regret_late']:.3f} | {r['slope']:+.4f} |")
    L.append("\n## Narrow-head sweep: selection regret vs head dim k (how narrow to fail)\n")
    L.append("| encoder | " + " | ".join(f"k={k}" for k in (2, 4, 8, 16, 32, 64)) + " |")
    L.append("|" + "---|" * 7)
    for r in res:
        if r["name"] == "oracle_attr":
            continue
        L.append(f"| {r['name']} | " + " | ".join(f"{r['ksweep'].get(k, float('nan')):.3f}" for k in (2, 4, 8, 16, 32, 64)) + " |")
    L.append("\ndrift_u ~ random => no drift-specific blind spot; slope~0 => not a drift effect; "
             "narrow k regret high & wide k low => threshold k* for the head.\n")
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
