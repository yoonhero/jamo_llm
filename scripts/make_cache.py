from transformers import AutoTokenizer
import json
import h5py
import tqdm
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("hg_tokenizer")

with open("./tmp/finetuning_dataset.json", "r") as f:
    data = json.load(f)

our_data = [
    {"instruction": "너를 누가 만들었니?", "input": "", "output": "세마고등학교 학생 연구팀이 개발한 GPT 아키텍처를 기반으로 학습된 자모입니다. 본 연구팀은 인공지능과 기계 학습 분야에서 저와 같은 소규모 언어 모델로 좋은 성능을 내는 것을 목표로 연구들을 진행하고 있습니다." },
    {"instruction": "넌 누구니?", "input": "", "output": "저는 자모라고 불리는 인공 지능 챗봇입니다. 세마고 학생들이 개발한 GPT 아키텍처를 기반으로 학습되었으며, 다양한 주제에 대한 대화를 지원하고 있습니다. 저에게 궁금한 것이 있으면 얼마든지 물어보세요!"},
    {"instruction": "자신을 소개해주세요.", "input": "", "output": "저는 자모라고 불리는 인공 지능 챗봇입니다. 세마고 학생들이 개발한 GPT 아키텍처를 기반으로 학습되었으며, 다양한 주제에 대한 대화를 지원하고 있습니다. 저에게 궁금한 것이 있으면 얼마든지 물어보세요!"},
]

data = data + our_data

PROMPT_DICT = {
    "prompt_input": (
        "요청을 적절히 완료하는 응답을 작성하세요.\n"
        "### 명령어:\n{instruction}\n\n### 입력:\n{input}\n\n### 응답:\n"
    ),
    "prompt_no_input": (
        "명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### 명령어:\n{instruction}\n\n### 응답:\n"
    ),
}

def _preprocess_hg(strings, tokenizer, block_size=256):
    kwargs = {"max_length": block_size + 1, "truncation": True, "padding": "max_length"}

    tokenized_list = []
    for text in tqdm.tqdm(strings):
        id = tokenizer.encode(f"<s> {text} </s>", **kwargs)
        tokenized_list.append(id)

    return tokenized_list

prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

sources = [
    prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    for example in data
]
targets = [f"{example['output']}" for example in data]

data = [source + target for source, target in zip(sources, targets)]

print(f"Sample Item:\n {data[0]}")

input_ids = _preprocess_hg(data, tokenizer)
np_ids = np.array(input_ids)

print(f"Sample Item:\n {np_ids[0]}")

total_dataset_size = np_ids.shape[0]
eval_size = 600
train_size = total_dataset_size - eval_size
print(f"Total Dataset: {total_dataset_size} | Train: {train_size} | Eval: {eval_size}")

np.random.shuffle(np_ids)
training_ds, eval_ds = np_ids[:train_size,:], np_ids[train_size:,:]

with h5py.File('./tmp/cache/sft-cache.hdf5', 'w') as f:
    f.create_dataset("train", data=training_ds, dtype=np.int16)
    f.create_dataset("eval", data=eval_ds, dtype=np.int16)


