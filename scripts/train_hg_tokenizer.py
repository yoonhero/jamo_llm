from tokenizers import ByteLevelBPETokenizer

paths = [""]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, vocab_size=8_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("../tokenizer", "jamo")

print(
    tokenizer.encode("<s> HI")
)