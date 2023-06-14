lines = []
with open("../tmp/10gb_corpus.txt", "r", buffering=1000000) as f:
    for line in f:
        text = line.strip()
        lines.append(text)

print(len(lines), lines[0])

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                 "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                 '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                 '∅': '', '³': '3', 'π': 'pi', }

def clean(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])

    return text.strip()


import re
import regex

def clean_str(text):
    clean(text, )
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'  # E-mail제거
    text = re.sub(pattern, '', text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'  # URL제거
    text = re.sub(pattern, '', text)
    # pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거

    # 일본어 문자를 식별하는 정규식 패턴입니다.
    japanese_pattern = r"[\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Han}]"
    text = regex.sub(japanese_pattern, "", text)
    pattern = "[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf]"
    clean_text = re.sub(pattern, "", text)
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[^\w\s\n]'  # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)
    text = re.sub('[-=+,#/\:^$.@*\"※~&%ㆍ』\\‘|\(\)\[\]\<\>`\'…》]', '', string=text)
    text = re.sub('\n', ' ', string=text)
    return text


from multiprocessing import Pool

pool = Pool(4)


def write_line(line):
    with open("../tmp/cleaned_chunk.txt", "a") as f:
        text = clean_str(line)
        f.write(text + "\n")


pool.imap_unordered(write_line, lines)
pool.close()
pool.join()