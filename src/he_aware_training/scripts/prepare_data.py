import os
import tqdm
import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pickle
from multiprocessing import Pool
from transformers import AutoTokenizer

from he_aware_training import PROJECT_ROOT


def get_dtype(vocab_size):
    if vocab_size < 65536:
        return np.uint16
    else:
        return np.uint32


def slice_val_from_train(data_dir, model_file_prefix, tokens_to_move=10_000_000):
    train_path = os.path.join(data_dir, f"{model_file_prefix}_train.bin")
    val_path = os.path.join(data_dir, f"{model_file_prefix}_val.bin")
    meta_path = os.path.join(data_dir, "meta.pkl")

    if not os.path.exists(train_path):
        print(f"Train file not found: {train_path}")
        return

    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if "uint32" in str(meta.get("dtype", "")):
            itemsize = 4
        else:
            itemsize = 2
    else:
        print("No meta.pkl found, assuming uint16 (2 bytes per token).")
        itemsize = 2

    bytes_to_move = tokens_to_move * itemsize
    assert bytes_to_move % itemsize == 0, "bytes_to_move must be a multiple of itemsize"
    file_size = os.path.getsize(train_path)

    if file_size < bytes_to_move:
        print("Train file too small to slice.")
        return

    print(f"Moving last {tokens_to_move/1e6:.1f}M tokens from Train to Val...")

    with open(train_path, "rb") as f_train:
        f_train.seek(-bytes_to_move, 2)
        data = f_train.read()

    with open(val_path, "wb") as f_val:
        f_val.write(data)
        f_val.flush()
        os.fsync(f_val.fileno())

    with open(train_path, "r+b") as f_train:
        f_train.seek(-bytes_to_move, 2)
        f_train.truncate()
        f_train.flush()
        os.fsync(f_train.fileno())

    print(f"Created {val_path}")
    print(f"Truncated {train_path} (removed {bytes_to_move / (1024**2):.2f} MB)")


_WORKER_TOK = None
_WORKER_EOS = None


def _init_worker(model_name, eos_id):
    global _WORKER_TOK, _WORKER_EOS
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOK = AutoTokenizer.from_pretrained(model_name)
    _WORKER_EOS = eos_id


def _tokenize_chunk(texts):
    encs = _WORKER_TOK(
        texts,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    out = []
    for ids in encs:
        out.extend(ids)
        if not ids or ids[-1] != _WORKER_EOS:
            out.append(_WORKER_EOS)
    return out


def _resolve_eos_id(tokenizer):
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        print("Warning: tokenizer has no EOS, using pad_token_id as EOS.")
        return tokenizer.pad_token_id
    print("Warning: tokenizer has no EOS or pad, falling back to 50256.")
    return 50256


def process_split(cfg, split_name, output_filename, tokenizer, num_workers):
    data_dir = os.path.join(PROJECT_ROOT, "data", cfg.dataset.name)
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, output_filename)

    print(f"Processing {split_name} -> {out_path} (workers={num_workers})...")

    dtype = get_dtype(tokenizer.vocab_size)
    print(f"Vocab size: {tokenizer.vocab_size}, using dtype: {dtype}")

    print(f"Streaming {cfg.dataset.hf_dataset.path}...")
    dataset = instantiate(cfg.dataset.hf_dataset, split=split_name, streaming=True)

    eos_id = _resolve_eos_id(tokenizer)
    model_name = cfg.model.name

    chunk_size = 1000
    flush_threshold = 1_000_000

    def text_chunks():
        chunk = []
        for example in dataset:
            text = (
                example.get("text", "")
                or example.get("content", "")
                or example.get("body", "")
                or example.get("article", "")
            )
            if not text:
                continue
            chunk.append(text)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    token_buffer = []
    total_tokens = 0
    total_texts = 0

    with open(out_path, "wb") as f, Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(model_name, eos_id),
    ) as pool:
        pbar = tqdm.tqdm(desc=f"Tokenizing {split_name}", unit=" texts")
        for tokens in pool.imap(_tokenize_chunk, text_chunks(), chunksize=4):
            token_buffer.extend(tokens)
            # one extra token (EOS) is appended per text; subtract it from the count
            n_texts = sum(1 for t in tokens if t == eos_id)
            total_texts += n_texts
            pbar.update(n_texts)

            if len(token_buffer) >= flush_threshold:
                arr = np.array(token_buffer, dtype=dtype)
                f.write(arr.tobytes())
                total_tokens += len(token_buffer)
                token_buffer = []

        if token_buffer:
            arr = np.array(token_buffer, dtype=dtype)
            f.write(arr.tobytes())
            total_tokens += len(token_buffer)
            token_buffer = []

        f.flush()
        os.fsync(f.fileno())
        pbar.close()

    print(f"Saved {total_tokens} tokens from {total_texts} texts to {out_path}")

    meta = {"vocab_size": tokenizer.vocab_size, "dtype": str(dtype)}
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


@hydra.main(
    version_base=None, config_path=f"{PROJECT_ROOT}/configs", config_name="prepare_data"
)
def main(cfg: DictConfig) -> None:
    model_name = cfg.model.name
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_file_prefix = model_name.split("/")[-1]

    # leave 1 core for the main process / HF streaming
    num_workers = max(1, (os.cpu_count() or 2) - 1)

    process_split(
        cfg, "train", f"{model_file_prefix}_train.bin", tokenizer, num_workers
    )

    data_dir = os.path.join(PROJECT_ROOT, "data", cfg.dataset.name)
    slice_val_from_train(data_dir, model_file_prefix, tokens_to_move=10_000_000)

    print(f"Done! Dataset ready in {data_dir}")


if __name__ == "__main__":
    main()
