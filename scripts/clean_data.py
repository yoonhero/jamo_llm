import re
import regex

lines = []

with open("../tmp/10gb_corpus.txt", "r", buffering=1000000) as f:
    for line in f:
        text = line.strip()
        lines.append(text)

print(len(lines))

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": "sqrt", "×": "x", "²": "2",
                 "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                 '∞': '무한', 'θ': '세타', '÷': '/', 'α': '알파', '•': '.', 'à': 'a', '−': '-', 'β': '베타',
                 '∅': '', '³': '3', 'π': '파이', }
def clean(text, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    specials = {'\u200b': '', '…': '.', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])

    return text.strip()

def clean_str(text):
    text = clean(text, punct_mapping)
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'  # E-mail제거
    text = re.sub(pattern, '', text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'  # URL제거
    text = re.sub(pattern, '', text)
    # 일본어 문자를 식별하는 정규식 패턴입니다.
    japanese_pattern = r"[\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Han}]"
    text = regex.sub(japanese_pattern, "", text)
    text = re.sub(pattern=pattern, repl='', string=text)
    text = re.sub('[-=+,#/\:^$.@*\"※~&%ㆍ』\\‘|\(\)\[\]\<\>`\'…》]', '', string=text)
    text = re.sub('\n', '', string=text)
    return text


from multiprocessing import Pool


def write_line(line):
    line = clean_str(line)
    cur = 0

    chunked = []
    if len(line) >= 512:
        for i in range(len(line)//512):
            chunked.append(line[i:512*i].strip())
            cur = 512 * i
    if cur < len(line):
        chunked.append(line[cur:].strip())

    with open("../tmp/cleaned_512.txt", "a") as f:   
        f.write("\n".join(chunked))

import tqdm
#for i in tqdm.tqdm(lines):
 #   write_line(i)

pool = Pool(5)
with tqdm.tqdm(total=len(lines)) as pbar:
    for _ in tqdm.tqdm(pool.imap_unordered(write_line, lines)):
        pbar.update()
    
#pool.imap_unordered(write_line, lines)
pool.close()
pool.join()
