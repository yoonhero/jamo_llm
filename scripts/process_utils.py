from xml.etree.ElementTree import parse
import json
import os
import re 
import glob
from multiprocessing import Pool

import tqdm
import gzip
from pathlib import Path


def preprocess_news(sentence):
    if "기자 =" in sentence:
        pos = sentence.index("기자 =") + 5
        sentence = sentence[pos:]
    elif "관련기사" in sentence or "참조링크" in sentence:
        sentence = ""
    if "무단 전재 및 재배포 금지" in sentence or "무단전재" in sentence or "저작권자ⓒ" in sentence: sentence = ""

    sentence = re.sub(r"(\w+)@\(이메일\)\sⓒ\s\(이메일\)", "", sentence)
    sentence = re.sub(r"(\w+)@", "", sentence)
    sentence = re.sub(r"[a-zA-Z0-9+-_.]@[a-zA-Z0-9-]\.[a-zA-Z0-9-.]", "", sentence)
    sentence = re.sub(r"\([^)]*\)", "", sentence)
    sentence = sentence.replace(",", "")

    return sentence

def read_text_from_xml(xml_dir:str):
    try:
        tree = parse(xml_dir)
        root = tree.getroot()
        text = " ".join([x.text for x in root.findall("text")[0].findall("p")])
        return text
    except: return ''


def read_text_from_txt(txt_dir: str, encoding):
    with open(txt_dir, "r", encoding=encoding) as f:
        texts = f.read()
    return texts


def process_c4(text):
    splitted_text = text.split("\n")
    
    copyright_marks = ["Copyright", "©"]
    not_essential = ["개인정보보호정책", "개인 정보 보호 정책"]
    for i, t in enumerate(splitted_text):
        if len(t) < 10: break 
