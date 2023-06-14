from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer, AutoTokenizer

def train_tokenizer():
    tokenizer = ByteLevelBPETokenizer(unicode_normalizer="nfkc", trim_offsets=True)

    # your text corpus data
    paths = [""]
    vocab_size = 8_000

    tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.save_model("./tokenizer", "jamo")

def proceess_tokenizer_file():
    transformers_gpt2_tokenizer = GPT2Tokenizer(
        vocab_file = 'jamo-vocab.json',
        merges_file = 'jamo-merges.txt',
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        add_special_tokens=True
    )

    transformers_gpt2_tokenizer.save_pretrained('hg_tokenizer')

if __name__ == "__main__":
    train_tokenizer()
    proceess_tokenizer_file()

    tokenizer = AutoTokenizer.from_pretrained("hg_tokenizer")