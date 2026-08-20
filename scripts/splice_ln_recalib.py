import argparse
import json
import os

import torch

PATCH_FIELDS = {                       # config field -> checkpoint buffer suffix
    "Ncoeffs": "inv_sqrt_approx.p",
    "Dcoeffs": "inv_sqrt_approx.q",
    "lin_alpha": "inv_sqrt_approx.lin_alpha",
    "lin_beta": "inv_sqrt_approx.lin_beta",
    "center_scale_sq": "center_scale_sq",
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="configs.json to copy and splice into")
    ap.add_argument("--calib", required=True, help="recalibrated configs.json to take domains from")
    ap.add_argument("--ckpt", required=True, help="checkpoint whose solver buffers get patched")
    ap.add_argument("--out-config", required=True, help="output config dir (writes configs.json)")
    ap.add_argument("--out-ckpt", required=True, help="output ckpt dir (writes last.pt)")
    ap.add_argument("--site", default="ALL",
                    help="one norm site to splice, or ALL (default)")
    args = ap.parse_args()

    base = json.load(open(args.base))
    new = json.load(open(args.calib))
    sites = list(base["norm"]) if args.site == "ALL" else [args.site]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]

    for s in sites:
        old_e, new_e = base["norm"][s], new["norm"][s]
        # the learned depth belongs to the checkpoint, not to the recalibration
        for k in ("gs_iters", "nr_iters"):
            new_e[k] = old_e[k]
        assert (len(new_e["Ncoeffs"]) == len(old_e["Ncoeffs"])
                and len(new_e["Dcoeffs"]) == len(old_e["Dcoeffs"])), (
            f"{s}: polynomial degree changed, that is a different circuit")
        print(f"  {s}: fit_hi {old_e['fit_hi']:.3f} -> {new_e['fit_hi']:.3f}"
              f"  (raw z_max {old_e['z_max']:.0f} -> {new_e['z_max']:.0f})")
        base["norm"][s] = new_e
        for field, suffix in PATCH_FIELDS.items():
            key = f"{s}.{suffix}"
            old_t = sd[key]
            sd[key] = torch.tensor(new_e[field], dtype=old_t.dtype).reshape(old_t.shape)

    os.makedirs(args.out_config, exist_ok=True)
    json.dump(base, open(f"{args.out_config}/configs.json", "w"), indent=2)
    print(f"config -> {args.out_config}/configs.json ({len(sites)} norm site(s) recalibrated)")

    os.makedirs(args.out_ckpt, exist_ok=True)
    torch.save(ck, f"{args.out_ckpt}/last.pt")
    print(f"ckpt   -> {args.out_ckpt}/last.pt "
          f"(iter_num {ck.get('iter_num')} kept, {len(PATCH_FIELDS) * len(sites)} buffers patched)")


if __name__ == "__main__":
    main()
