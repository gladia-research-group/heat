import math

import torch
import torch.nn.functional as F
import wandb


@torch.no_grad()
def evaluate_dataset(
    cfg,
    model,
    data_loader_fn,
    device,
    eval_iters=200,
    desc="val",
    output_attentions=True,
    log_to_wandb=True,
    verbose=True,
):
    """
    Evaluates the model on the dataset and returns Loss and Perplexity.

    Args:
        log_to_wandb: If True, logs metrics to wandb. Set False to handle logging externally.
    """
    model.eval()
    model.to(device)
    losses = torch.zeros(eval_iters)

    for i in range(eval_iters):
        X, Y = data_loader_fn(cfg, split="val")
        if X is None:
            break

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(X, labels=Y, output_attentions=output_attentions)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[1]

        losses[i] = loss.item()

    mean_loss = losses.mean().item()
    perplexity = math.exp(mean_loss)

    if log_to_wandb:
        wandb.log(
            {
                f"{desc}/loss": mean_loss,
                f"{desc}/perplexity": perplexity,
            }
        )

    if verbose:
        print(
            f"📊 [Evaluation] {desc.capitalize()} Loss: {mean_loss:.4f}, Perplexity: {perplexity:.2f}"
        )

    return mean_loss, perplexity


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=30,
    temperature=0.5,
    top_k=10,
    repetition_penalty=1.5,
    device="cuda",
):
    model.eval()

    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    idx = tokens

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]

            outputs = model(idx_cond)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0]

            logits = logits[:, -1, :]

            logits = logits / temperature

            if repetition_penalty > 1.0:
                score = torch.gather(logits, 1, idx)
                score = torch.where(
                    score < 0, score * repetition_penalty, score / repetition_penalty
                )

                logits.scatter_(1, idx, score)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            probs = F.softmax(logits, dim=-1)
            probs = torch.clamp(probs, min=0.0)
            # Re-normalize in case clamping broke the distribution
            probs = probs / probs.sum(dim=-1, keepdim=True)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if idx_next.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(idx[0].tolist())
    return generated_text
