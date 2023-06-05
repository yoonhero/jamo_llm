import torch
from pathlib import Path

from jamo import JAMO, Tokenizer
import utils

@torch.no_grad()
def generate(
    model: JAMO,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    max_seq_length=None,
    temperature: float = 1.0,
    top_k=None,
    eos_id=None,
) -> torch.Tensor:
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    T_new = T + max_new_tokens
    if max_seq_length is None:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    # generate max_new_tokens tokens
    for _ in range(max_new_tokens):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, max_seq_length, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx

if __name__ == "__main__":
    model, _, _ = utils.load_model(Path("./tmp/checkpoint"), 0.0)
    tokenizer = Tokenizer("./tokenizer/corpus.model")

    token = tokenizer.encode("", bos=True)
    token = torch.tensor([token], dtype=torch.long, device="cuda")
    output = generate(model, token, max_new_tokens=100, temperature=0.8, top_k=4, eos_id=tokenizer.encode("</s>")[0])
    print(output)
    result = tokenizer.decode(output)

    print(result)