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
parser.add_argument("--clean", action="store_true")

args = parser.parse_args()

lines = []

path = args.path

with open(path, "r", buffering=1000000) as f:
    for line in f:
        text = line.strip()
        lines.append(text)

cleaning = args.clean
result_dir = args.result_dir

def write_line(line):
    if cleaning:
        line = TextPreprocessing.preprocess(line)
    prev_cur = 0
    chunked = []
    if len(line) >= 512:
        for i in range(len(line) // 512):
            start_cur = i*512
            for space in range(10):
                if line[start_cur+space] == " ":
                    break
            start_cur += space
            chunked.append(line[prev_cur:start_cur].strip())
            prev_cur = start_cur
    if prev_cur < len(line) and len(line) - prev_cur > 200:
        chunked.append(line[prev_cur:].strip())

    with open(result_dir, "a", encoding="utf-8") as f:
        f.write("\n".join(chunked))


pool = Pool(os.cpu_count()-1)
with tqdm.tqdm(total=len(lines)) as pbar:
   for _ in tqdm.tqdm(pool.imap_unordered(write_line, lines)):
       pbar.update()

pool.close()
pool.join()
