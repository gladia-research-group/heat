import json

import torch
import torch.nn as nn

from transformers.activations import GELUActivation, NewGELUActivation
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.vit.modeling_vit import ViTAttention

from he_aware_training.modules.learnable_layers import (
    LearnedHEGeLU,
    THORCompositeGeLU,
    LearnedLayerNormHE,
)

from he_aware_training.modules.learnable_enc_llm_layers import (
    EncLLMLearnedGeLU,
    EncLLMLearnedHEAttention,
)

from he_aware_training.modules.learnable_thor_layers import (
    THORLearnableHEAttention,
)

from he_aware_training.modules.learnable_vit_layers import (
    THORLearnableViTAttention,
)

from he_aware_training.modules.learnable_bert_layers import (
    THORLearnableBertAttention,
)


_ATTENTION_REGISTRY = {
    "enc_llm": EncLLMLearnedHEAttention,
    "thor": THORLearnableHEAttention,
    "thor_vit": THORLearnableViTAttention,   
    "thor_bert": THORLearnableBertAttention, 
}

_ATTENTION_TARGET = {
    "enc_llm": GPT2Attention,
    "thor": GPT2Attention,
    "thor_vit": ViTAttention,
    "thor_bert": BertSelfAttention,
}

_GELU_REGISTRY = {
    "sign_poly": EncLLMLearnedGeLU,
    "softsign":  LearnedHEGeLU,
    "chebyshev": LearnedHEGeLU,   
    "thor":      THORCompositeGeLU,  
}


def _replace_layers(module, policy, verbose=False, on_construct=None, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        replaced = False
        for target_cls, info in policy.items():
            if info is None:
                continue
            if not isinstance(child, target_cls):
                continue

            replacement_cls = info["cls"]
            arg_extractor = info["args"]
            extra_args = info.get("extra_args", [])
            from_layer_kwargs_fn = info.get("from_layer_kwargs_fn", None)
            pre_replace_fn = info.get("pre_replace_fn", None)

            if verbose:
                print(
                    f"   -> Replacing {full_name} ({child.__class__.__name__}) "
                    f"with {replacement_cls.__name__}"
                )

            try:
                if pre_replace_fn:
                    pre_replace_fn(full_name, child)

                if on_construct is not None:
                    on_construct(full_name, target_cls)

                args = list(extra_args) + arg_extractor(child)
                new_layer = replacement_cls(*args)

                if hasattr(new_layer, "from_layer"):
                    from_layer_kwargs = (
                        from_layer_kwargs_fn(child, target_cls)
                        if from_layer_kwargs_fn
                        else {}
                    )
                    new_layer.from_layer(child, **from_layer_kwargs)

                setattr(module, name, new_layer)
                replaced = True
                break
            except Exception as e:
                print(f"      [!] Failed to replace {full_name}: {e}")
                import traceback

                traceback.print_exc()

        if not replaced:
            _replace_layers(child, policy, verbose, on_construct, full_name)

_CALIB_SECTION = {"thor": "softmax", "ln": "norm", "gelu": "softgelu", "softgelu": "softgelu"}


def _load_calib_lookup(path, component=None):
    if path is None:
        return {}, None
    with open(path) as f:
        calib = json.load(f)
    section = _CALIB_SECTION.get(component)
    if (section and isinstance(calib, dict)
            and any(k in calib for k in ("norm", "softmax", "softgelu"))):
        calib = {"per_layer": calib.get(section, {})}
    if isinstance(calib, dict) and "global" in calib:
        return {}, calib["global"]
    if isinstance(calib, dict) and "per_layer" in calib:
        per_layer = calib["per_layer"]
        def _v5(k):
            return (k.replace(".encoder.layer.", ".layers.")
                     .replace(".intermediate.intermediate_act_fn", ".mlp.activation_fn"))
        def _v4(k):
            return (k.replace(".layers.", ".encoder.layer.")
                     .replace(".mlp.activation_fn", ".intermediate.intermediate_act_fn"))
        for k in list(per_layer):
            per_layer.setdefault(_v5(k), per_layer[k])
            per_layer.setdefault(_v4(k), per_layer[k])
        return per_layer, None
    return calib, None  # legacy flat per-layer format


def install_gelu_input_guards(model, calib_path, factor=0.95, strength=8.0, component="softgelu"):
    from he_aware_training.utils.utils import squeeze

    try:
        per_layer, _ = _load_calib_lookup(calib_path, component)
    except (OSError, ValueError, TypeError) as e:
        print(f"[gelu-guard] calib map unavailable ({calib_path}: {e}) — guards NOT installed")
        return 0
    gelu_types = (nn.GELU, GELUActivation, NewGELUActivation)
    n = 0
    for name, module in model.named_modules():
        entry = per_layer.get(name)
        if entry is None or "xmax" not in entry or not isinstance(module, gelu_types):
            continue
        bound = factor * float(entry["xmax"])

        def pre(mod, args, _b=bound):
            if not mod.training:
                return None
            return (squeeze(args[0], -_b, _b, strength=strength, tag="gelu.x"),) + args[1:]

        module.register_forward_pre_hook(pre)
        print(f"[gelu-guard] {name}: training clamp |x| <= {bound:.2f}")
        n += 1
    return n


def _inject_namespace(approx_config, ns, entry):
    from omegaconf import OmegaConf
    target = approx_config[ns]
    target_keys = set(target.keys())
    for key, val in entry.items():
        if key in target_keys:
            OmegaConf.update(target, key, val)


def perform_he_surgery(model, cfg, verbose=False):
    approx_name = cfg.model.approximation.name
    approx_config = cfg.model.approximation[approx_name]

    attention_kind = approx_config.attention_kind
    if attention_kind not in _ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention_kind {attention_kind!r}; "
            f"expected one of {sorted(_ATTENTION_REGISTRY)}"
        )
    attention_cls = _ATTENTION_REGISTRY[attention_kind]
    attention_target = _ATTENTION_TARGET[attention_kind]
    if attention_target is ViTAttention:
        attention_args = lambda x: [getattr(x, "config", None) or x.attention.config, None]
    else:
        attention_args = lambda x: [x.config, getattr(x, "layer_idx", None)]
    if verbose:
        print(f"   [attention] kind={attention_kind} → {attention_cls.__name__} "
              f"(target {attention_target.__name__})")

    gelu_kind = getattr(approx_config, "gelu_kind", "sign_poly")
    if gelu_kind not in _GELU_REGISTRY:
        raise ValueError(
            f"Unknown gelu_kind {gelu_kind!r}; expected one of {sorted(_GELU_REGISTRY)}"
        )
    gelu_cls = _GELU_REGISTRY[gelu_kind]
    gelu_ns = "gelu" if gelu_kind == "thor" else "softgelu"   # THOR has its own config namespace
    if verbose:
        print(f"   [gelu] kind={gelu_kind} → {gelu_cls.__name__} (ns={gelu_ns})")

    if gelu_kind in ("softsign", "chebyshev", "thor"):
        gelu_args = lambda x: []
        gelu_extra = [approx_config]
    else:
        gelu_args = lambda x: []
        gelu_extra = []

    thor_per_layer, thor_global = _load_calib_lookup(
        approx_config.thor.calib_path if "thor" in approx_config else None, "thor"
    )
    ln_per_layer, ln_global = _load_calib_lookup(
        approx_config.ln.calib_path if "ln" in approx_config else None, "ln"
    )
    sg_per_layer, sg_global = _load_calib_lookup(
        approx_config[gelu_ns].calib_path if gelu_ns in approx_config else None, gelu_ns
    )

    def on_construct(full_name, target_cls):
        if target_cls in (GPT2Attention, ViTAttention, BertSelfAttention):
            entry = thor_global if thor_global is not None else thor_per_layer.get(full_name)
            if entry is not None:
                _inject_namespace(approx_config, "thor", entry)
        elif target_cls is nn.LayerNorm:
            entry = ln_global if ln_global is not None else ln_per_layer.get(full_name)
            if entry is not None:
                _inject_namespace(approx_config, "ln", entry)
        elif target_cls in (nn.GELU, NewGELUActivation, GELUActivation):
            entry = sg_global if sg_global is not None else sg_per_layer.get(full_name)
            if entry is not None:
                _inject_namespace(approx_config, gelu_ns, entry)

    policy = {
        nn.LayerNorm: (
            {
                "cls": LearnedLayerNormHE,
                "args": lambda x: [x.weight.shape[0]],
                "extra_args": [approx_config],
            }
            if cfg.model.surgery.norm
            else None
        ),
        nn.GELU: (
            {"cls": gelu_cls, "args": gelu_args, "extra_args": gelu_extra}
            if cfg.model.surgery.activation
            else None
        ),
        NewGELUActivation: (
            {"cls": gelu_cls, "args": gelu_args, "extra_args": gelu_extra}
            if cfg.model.surgery.activation
            else None
        ),
        GELUActivation: (
            {"cls": gelu_cls, "args": gelu_args, "extra_args": gelu_extra}
            if cfg.model.surgery.activation
            else None
        ),
        attention_target: (
            {
                "cls": attention_cls,
                "args": attention_args,
                "extra_args": [approx_config],
            }
            if cfg.model.surgery.attention
            else None
        ),
    }

    _replace_layers(model, policy, verbose, on_construct=on_construct)

    model.to(dtype=torch.float32)
    return model


def audit_surgery(model, cfg, expected=None, strict=True):
    approx_name = cfg.model.approximation.name
    approx_config = cfg.model.approximation[approx_name]
    attention_kind = approx_config.attention_kind
    learn_attn_cls = _ATTENTION_REGISTRY[attention_kind]
    orig_attn_cls = _ATTENTION_TARGET[attention_kind]

    from collections import Counter
    counts = Counter(type(m).__name__ for _, m in model.named_modules())

    obs = {
        "attention": counts.get(learn_attn_cls.__name__, 0),
        "ln": counts.get(LearnedLayerNormHE.__name__, 0),
        "gelu": (counts.get(THORCompositeGeLU.__name__, 0)
                 + counts.get(LearnedHEGeLU.__name__, 0)
                 + counts.get(EncLLMLearnedGeLU.__name__, 0)),
    }

    leftover = {
        "attention": counts.get(orig_attn_cls.__name__, 0),
        "ln": counts.get("LayerNorm", 0),
        "gelu": (counts.get("GELUActivation", 0)
                 + counts.get("NewGELUActivation", 0) + counts.get("GELU", 0)),
    }

    print(f"[surgery-audit] kind={attention_kind} learnable={learn_attn_cls.__name__} "
          f"target={orig_attn_cls.__name__}")
    print(f"[surgery-audit] replaced: attention={obs['attention']} ln={obs['ln']} "
          f"gelu={obs['gelu']}  |  leftover HF: attention={leftover['attention']} "
          f"ln={leftover['ln']} gelu={leftover['gelu']}")

    errs = []
    if cfg.model.surgery.attention and obs["attention"] == 0:
        errs.append(f"ATTENTION NOT REPLACED: 0 {learn_attn_cls.__name__} found — "
                    f"{orig_attn_cls.__name__} target likely not present/registered "
                    f"({leftover['attention']} original blocks survived).")
    if cfg.model.surgery.attention and leftover["attention"]:
        errs.append(f"{leftover['attention']} original {orig_attn_cls.__name__} "
                    f"survived surgery (partial replacement).")
    if cfg.model.surgery.norm and leftover["ln"]:
        errs.append(f"{leftover['ln']} raw nn.LayerNorm survived surgery.")
    if cfg.model.surgery.activation and leftover["gelu"]:
        errs.append(f"{leftover['gelu']} raw GELU survived surgery.")
    if expected:
        for site, want in expected.items():
            if obs.get(site) != want:
                errs.append(f"{site}: expected {want}, got {obs.get(site)}.")

    if errs and strict:
        raise ValueError("[surgery-audit] FAILED:\n  - " + "\n  - ".join(errs))
    for e in errs:
        print(f"[surgery-audit] WARN: {e}")
    return obs




