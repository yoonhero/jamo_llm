import sentencepiece as spm

corpus = "../tmp/10gb_corpus.txt"
prefix = "17000_tokenizer"
vocab_size = 16993
spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7} --train_extremely_large_corpus=true" + 
    " --model_type=bpe" +
    " --max_sentence_length=99999" + # 문장 최대 길이
    " --pad_id=0 --pad_piece=<p>" + # pad (0)
    " --unk_id=1 --unk_piece=<unk>" + # unknown (1)
    " --bos_id=2 --bos_piece=<s>" + # begin of sequence (2)
    " --eos_id=3 --eos_piece=</s>" # end of sequence (3)
)
