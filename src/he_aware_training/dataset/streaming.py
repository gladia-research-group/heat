import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
from hydra.utils import instantiate
from transformers import AutoTokenizer

from he_aware_training.utils.models import infer_seq_length


class StreamingDataset(IterableDataset):
    def __init__(self, dataset_cfg, tokenizer, split, model_name):
        self.tokenizer = tokenizer
        self.eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id

        if self.eos_id is None:
            raise ValueError(
                "Tokenizer must have an eos_token_id or pad_token_id defined."
            )

        self.block_size = infer_seq_length(model_name)

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        print(
            f"[Dataset] Rank {self.rank}/{self.world_size}: Initializing {dataset_cfg.path}"
        )

        self.hf_dataset = instantiate(dataset_cfg, split=split, streaming=True)

        try:
            self.hf_dataset = self.hf_dataset.shard(
                num_shards=self.world_size, index=self.rank
            )
            if self.rank == 0:
                print(
                    f"[Dataset] Successfully sharded dataset into {self.world_size} parts."
                )
        except Exception as e:
            print(
                f"[Dataset] Warning: .shard() failed ({e}). Falling back to seed manipulation."
            )

        self.hf_dataset = self.hf_dataset.shuffle(
            seed=42 + self.rank, buffer_size=10000
        )

    def __iter__(self):
        worker_info = get_worker_info()
        dataset_iter = self.hf_dataset

        if worker_info is not None:
            pass

        buffer = []
        iterator = iter(dataset_iter)

        for sample in iterator:
            text = sample.get(
                "text",
                sample.get(
                    "content", sample.get("abstract", sample.get("article", ""))
                ),
            )

            if not text:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            buffer.append(self.eos_id)

            need = self.block_size + 1
            while len(buffer) >= need:
                chunk = torch.tensor(buffer[:need], dtype=torch.long)
                yield chunk[:-1], chunk[1:]
                buffer = buffer[need:]
