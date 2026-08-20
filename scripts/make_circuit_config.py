import argparse
import json
import os
import re
import sys

def _calib_file(calib):
    """Accept a dir (merged configs.json inside, or the retired thor/ln/gelu split) or a file."""
    import os
    if os.path.isfile(calib):
        return calib
    merged = os.path.join(calib, "configs.json")
    if os.path.isfile(merged):
        return merged
    raise SystemExit(f"no configs.json in {calib} (and the split form has been retired)")


import torch
import torch.nn.functional as F


def _remap_gpt2(k):
    return k


def _remap_vit(k):
    k = re.sub(r"^vit\.layers\.", "vit.encoder.layer.", k)
    k = re.sub(r"\.mlp\.activation_fn$", ".intermediate.intermediate_act_fn", k)
    k = re.sub(r"^vit\.layernorm$", "transformer.ln_f", k)
    k = re.sub(r"^vit\.encoder\.layer\.(\d+)\.layernorm_before$", r"transformer.h.\1.ln_1", k)
    k = re.sub(r"^vit\.encoder\.layer\.(\d+)\.layernorm_after$", r"transformer.h.\1.ln_2", k)
    k = re.sub(r"^vit\.encoder\.layer\.(\d+)\.attention$", r"transformer.h.\1.attn", k)
    k = re.sub(r"^vit\.encoder\.layer\.(\d+)\.intermediate\.intermediate_act_fn$",
               r"transformer.h.\1.mlp.act", k)
    return k


def _remap_bert(k):
    k = re.sub(r"^bert\.layers\.", "bert.encoder.layer.", k)
    k = re.sub(r"\.mlp\.activation_fn$", ".intermediate.intermediate_act_fn", k)
    k = re.sub(r"^bert\.embeddings\.LayerNorm$", "embeddings.LayerNorm", k)
    k = re.sub(r"^bert\.encoder\.layer\.(\d+)\.attention\.output\.LayerNorm$",
               r"transformer.h.\1.ln_1", k)
    k = re.sub(r"^bert\.encoder\.layer\.(\d+)\.output\.LayerNorm$", r"transformer.h.\1.ln_2", k)
    k = re.sub(r"^bert\.encoder\.layer\.(\d+)\.attention\.self$", r"transformer.h.\1.attn", k)
    k = re.sub(r"^bert\.encoder\.layer\.(\d+)\.intermediate\.intermediate_act_fn$",
               r"transformer.h.\1.mlp.act", k)
    return k


DIMS = {"n_layers": 12, "n_embd": 768, "n_head": 12, "n_inner": 3072}
ARCHS = {
    "gpt2": dict(remap=_remap_gpt2, base="clone", counts="ckpt", needs=("src",)),
    "vit": dict(remap=_remap_vit, base="assemble", counts="config", needs=("calib", "counts_src")),
    "bert": dict(remap=_remap_bert, base="assemble", counts="forward", needs=("calib",)),
}

def counts_from_ckpt(ckpt, remap):
    def min_of(logits):
        n = 0
        for x in logits.tolist():
            if x != -10.0:
                break
            n += 1
        return n

    def mode_of(logits, min_iter, temp):
        l = logits[min_iter:] / temp
        lh, lc = F.logsigmoid(l), F.logsigmoid(-l)
        p = torch.exp(lh + torch.cumsum(lc, 0) - lc)
        p = torch.cat([p, (1 - p.sum()).clamp_min(0).unsqueeze(0)])
        return min_iter + int(p.argmax())

    sd = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)["model_state_dict"]
    out = {}
    for k, v in sd.items():
        if not k.endswith("halt_logits"):
            continue
        parts = k.split(".")
        solver, site = parts[-2], ".".join(parts[:-3])
        mi = min_of(v)
        temp = float(sd.get(k.replace("halt_logits", "_halt_temp"), torch.tensor(1.0)))
        m = mode_of(v, mi, temp)
        if "softmax" in k:
            out[(remap(site), "init" if solver.endswith("init") else "refine")] = m
        else:
            gold = solver == "ponder_goldschmidt"
            out[(remap(site), "gs" if gold else "nr")] = m if gold else m - mi
    return out


def counts_from_config(path):
    c = json.load(open(os.path.join(path, "configs.json")))
    out = {}
    for site, e in c["norm"].items():
        out[(site, "gs")], out[(site, "nr")] = e["gs_iters"], e["nr_iters"]
    for site, e in c["softmax"].items():
        out[(site, "init")] = e["gs_iters_scaled"]
        out[(site, "refine")] = e["gs_iters_refine_scaled"]
    return out


def counts_from_forward(ckpt, calib, root, remap):
    """Surgery the model, load the ckpt, one forward pass to populate p_dist, read the
    modes off it. The only source independent of the halt-prefix convention."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from he_aware_training.utils.loss import DepthRegularizer
    from he_aware_training.utils.surgery import perform_he_surgery

    with initialize_config_dir(version_base=None, config_dir=f"{root}/configs"):
        cfg = compose(config_name="he_aware_train_bert", overrides=[
            "device=cpu",
            f"model.approximation.hybrid.thor.calib_path={_calib_file(calib)}",
            f"model.approximation.hybrid.ln.calib_path={_calib_file(calib)}",
            f"model.approximation.hybrid.gelu.calib_path={_calib_file(calib)}",
        ])
    model = perform_he_surgery(instantiate(cfg.model.instance), cfg, verbose=False)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    model.eval()

    dreg = DepthRegularizer(model, lambda_depth=0.0, scheduler=None)
    with torch.no_grad():
        model(torch.tensor([[101, 2023, 3185, 2003, 2307, 102]], dtype=torch.long))
    stats = dreg.get_eval_stats()
    dreg.close()

    tail = {"ponder_goldschmidt": "gs", "ponder_newton": "nr",
            "ponder_goldschmidt_init": "init", "ponder_goldschmidt_refine": "refine"}
    out = {}
    for key, n in stats.items():
        if not key.startswith("depth_mode/"):
            continue
        k = key[len("depth_mode/"):-len("_mode")]
        for suf, solver in tail.items():
            if k.endswith("." + suf):
                stem = k[: -len("." + suf)]
                stem = stem.rsplit(".inv_sqrt_approx", 1)[0].rsplit(".softmax", 1)[0]
                out[(remap(stem), solver)] = int(n)
                break
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS))
    ap.add_argument("--ckpt", required=True, help="trained checkpoint (.pt)")
    ap.add_argument("--dst", required=True, help="output configs.json, or a dir to write it into")
    ap.add_argument("--mirror", help="also write the same config here (e.g. the training tree), "
                                     "so both trees hold the same pipeline stage")
    ap.add_argument("--src", help="[gpt2] configs.json to clone")
    ap.add_argument("--calib", help="[vit/bert] calibration: a configs.json, or a dir containing one")
    ap.add_argument("--counts-src", help="[vit] config dir holding the verified mode counts")
    ap.add_argument("--root", default=os.environ.get("PROJECT_ROOT", "."),
                    help="repo root for relative paths (default $PROJECT_ROOT or '.')")
    args = ap.parse_args()

    arch = ARCHS[args.arch]
    for need in arch["needs"]:
        if not getattr(args, need):
            ap.error(f"--arch {args.arch} requires --{need.replace('_', '-')}")
    remap = arch["remap"]

    if arch["counts"] == "ckpt":
        counts = counts_from_ckpt(args.ckpt, remap)
    elif arch["counts"] == "config":
        counts = counts_from_config(args.counts_src)
    else:
        counts = counts_from_forward(args.ckpt, args.calib, args.root, remap)

    if arch["base"] == "clone":
        cfg = json.load(open(args.src))
    else:
        if os.path.isdir(args.calib):
            per = {n: json.load(open(f"{args.calib}/{n}.json"))["per_layer"]
                   for n in ("thor", "ln", "gelu")}
        else:
            whole = json.load(open(args.calib))
            per = {"thor": whole["softmax"], "ln": whole["norm"],
                   "gelu": whole["softgelu"]}
        cfg = {"model": dict(DIMS), "softgelu": {}, "norm": {}, "softmax": {}}
        for section, src in (("softgelu", "gelu"), ("norm", "ln"), ("softmax", "thor")):
            for k, e in per[src].items():
                cfg[section][remap(k)] = dict(e)

    n = 0
    for site, e in cfg["norm"].items():
        for solver, field in (("gs", "gs_iters"), ("nr", "nr_iters")):
            if (site, solver) in counts:
                e[field] = counts[(site, solver)]
                n += 1
    for site, e in cfg["softmax"].items():
        if (site, "init") in counts:
            e["gs_iters_scaled"] = counts[(site, "init")]
            n += 1
        if (site, "refine") in counts:
            r = counts[(site, "refine")]
            e["gs_iters_refine_scaled"] = r
            passes = e.get("refinement_iters", len(e.get("per_step_refine_iters", [1])))
            e["per_step_refine_iters"] = [r] * passes
            n += 1

    ln_gs = sum(v["gs_iters"] for v in cfg["norm"].values())
    ln_nr = sum(v["nr_iters"] for v in cfg["norm"].values())
    sm_i = sum(v["gs_iters_scaled"] for v in cfg["softmax"].values())
    sm_x = sum(sum(v["per_step_refine_iters"]) for v in cfg["softmax"].values())

    out = args.dst if args.dst.endswith(".json") else os.path.join(args.dst, "configs.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(cfg, open(out, "w"), indent=1)
    print(f"wrote {out}: {n} counts")
    if args.mirror:
        mirror = os.path.join(args.mirror, "configs.json") if not args.mirror.endswith(".json") else args.mirror
        os.makedirs(os.path.dirname(mirror) or ".", exist_ok=True)
        json.dump(cfg, open(mirror, "w"), indent=1)
        print(f"mirrored -> {mirror}")
    print(f"  LN gs {ln_gs} + nr {ln_nr} + sm init {sm_i} + refine {sm_x} "
          f"= {ln_gs + ln_nr + sm_i + sm_x} it/fwd executed")


if __name__ == "__main__":
    main()
