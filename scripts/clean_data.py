import re
import regex
import tqdm
from multiprocessing import Pool
import argparse
import os

from text_preprocessing import TextPreprocessing

parser = argparse.ArgumentParser(description='Chunking the Dataset with Text-Processing')
parser.add_argument("--path", type=str, default="../tmp/corpus.txt")
parser.add_argument("--result_dir", type=str, default="../tmp/cleaned_512.txt")
parser.add_argument("--clean", type=bool, action="store_true")

args = parser.parse_args()

def loading_corpus(path):
    lines = []

    with open(path, "r", buffering=1000000) as f:
        for line in f:
            text = line.strip()
            lines.append(text)

    return lines

lines = loading_corpus(args.path)
print(len(lines))

cleaning = args.clean
result_dir = args.result_dir

def write_line(line):
    if cleaning:
        line = TextPreprocessing.preprocess(line)

    cur = 0
    chunked = []
    if len(line) >= 512:
        for i in range(len(line) // 512):
            chunked.append(line[i:512 * i].strip())
            cur = 512 * i
    if cur < len(line):
        chunked.append(line[cur:].strip())

    with open(result_dir, "a") as f:
        f.write("\n".join(chunked))


pool = Pool(os.cpu_count() - 1)
with tqdm.tqdm(total=len(lines)) as pbar:
    for _ in pool.imap_unordered(write_line, lines):
        pbar.update()

pool.close()
pool.join()