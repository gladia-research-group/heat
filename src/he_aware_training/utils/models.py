from transformers import AutoConfig


def infer_seq_length(model_name):
    """
    Determines the optimal sequence length by checking config attributes.
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    derived_len = None
    candidates = [
        "max_position_embeddings",
        "seq_length",
        "n_positions",
        "n_ctx",
        "sliding_window",
    ]

    for attr in candidates:
        val = getattr(config, attr, None)
        if val is not None and isinstance(val, int):
            derived_len = val
            break

    if derived_len is None:
        print("Falling back to default sequence length of 1024.")
        derived_len = 1024  # fallback

    return derived_len
