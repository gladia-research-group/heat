import os

import torch


def compose_get_sst2_batch(model_name, device, micro_batch, val_size=2000,
                           max_len=66, min_len=3, seed=1337,
                           long_frac=0.0, long_len=128,
                           hf_path="nyu-mll/glue", hf_config="sst2",
                           split="train", text_field="sentence",
                           label_field="label"):
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(model_name)

    train = load_dataset(hf_path, hf_config, split=split)

    enc = tok(list(train[text_field]), truncation=True, max_length=max_len)["input_ids"]
    labels = list(train[label_field])

    n = len(enc)
    splits = {"train": range(0, n - val_size), "val": range(n - val_size, n)}
    buckets = {}
    for split, idxs in splits.items():
        by_len = {}
        for i in idxs:
            ids = enc[i]
            if len(ids) < min_len:
                continue
            by_len.setdefault(len(ids), []).append(i)
        buckets[split] = [
            (torch.tensor([enc[i] for i in idxs_l], dtype=torch.long),
             torch.tensor([labels[i] for i in idxs_l], dtype=torch.long))
            for L, idxs_l in sorted(by_len.items())
        ]
    weights = {s: torch.tensor([float(x.shape[0]) for x, _ in b])
               for s, b in buckets.items()}
    gen = torch.Generator().manual_seed(seed)
    counts = {s: sum(int(x.shape[0]) for x, _ in b) for s, b in buckets.items()}
    print(f"[sst2] train sentences: {counts['train']} ({len(buckets['train'])} length buckets), "
          f"val (held-out train slice): {counts['val']} ({len(buckets['val'])} buckets)")

    long_rows = None
    if long_frac > 0:
        sep, cls = tok.sep_token_id, tok.cls_token_id
        stream = []
        for i in splits["train"]:
            stream.extend(tok(train[text_field][i], add_special_tokens=False)["input_ids"])
            stream.append(sep)
        body = long_len - 2
        n_rows = len(stream) // body
        long_rows = torch.tensor(
            [[cls] + stream[r * body:(r + 1) * body] + [sep] for r in range(n_rows)],
            dtype=torch.long)
        print(f"[sst2] long-row arm: {n_rows} rows of {long_len} tokens, p={long_frac}")

    def get_batch(split):
        if split == "train" and long_rows is not None and \
                torch.rand(1, generator=gen).item() < long_frac:
            ix = torch.randperm(long_rows.shape[0], generator=gen)[:micro_batch]
            X = long_rows[ix]
            Y = torch.zeros(len(ix), dtype=torch.long)   # dummy — KD ignores labels
            if "cuda" in str(device):
                return (X.pin_memory().to(device, non_blocking=True),
                        Y.pin_memory().to(device, non_blocking=True))
            return X.to(device), Y.to(device)
        b = buckets["train" if split == "train" else "val"]
        w = weights["train" if split == "train" else "val"]
        bi = int(torch.multinomial(w, 1, generator=gen).item())
        X_all, Y_all = b[bi]
        n_avail = X_all.shape[0]
        take = min(micro_batch, n_avail)
        ix = torch.randperm(n_avail, generator=gen)[:take]
        X, Y = X_all[ix], Y_all[ix]
        if "cuda" in str(device):
            X = X.pin_memory().to(device, non_blocking=True)
            Y = Y.pin_memory().to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)
        return X, Y

    meta = {"num_labels": 2, "n_train": counts["train"], "n_val": counts["val"]}
    return get_batch, meta


def load_bert_for_heat(model_dir, num_labels=2):
    from transformers import BertForSequenceClassification

    model = BertForSequenceClassification.from_pretrained(
        model_dir, num_labels=int(num_labels), attn_implementation="eager")
    return model
