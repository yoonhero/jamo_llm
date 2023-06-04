import json
import os
import re 
import glob
from multiprocessing import Pool
import tqdm
import gzip
from pathlib import Path
import os
import re
import json
import zipfile
import pandas as pd 

from process_utils import read_text_from_txt, read_text_from_xml, preprocess_news

cleaning_first_patterns = [
  r"\[\*[^\]]+\]",
  r"~~[^~]+~~"
]
cleaning_first_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in cleaning_first_patterns]

cleaning_patterns = [
  r"\([^\)]+\)",
  r'\[[^]]*\]'
]
cleaning_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in cleaning_patterns]

# \n -> 띄어쓰기 
# \' -> '
replace_patterns = {
    '\\n': "\n",
    "\\'": "'"
}


def clean_text(text):
  for regex in cleaning_first_patterns:
    text = re.sub(regex, "", text)
  for regex in cleaning_patterns:
    text = re.sub(regex, "", text)
  for k, v in replace_patterns:
    text = text.replace(k, v)
  return text


cpu_cores = os.cpu_count() 


class Preprocess():
    def __init__(self, dataset_folder: Path):
        self.result_dir = Path("../../dataset/processed/corpus.txt")
        self.result_dir.parent.mkdir(exist_ok=True)

        self.target_files = glob.glob(str(dataset_folder/"*"))
        self.total_files = len(self.target_files)

    def read(self):
        return NotImplemented

    def write(self, text):
        mode = "a" if self.result_dir.exists() else "w"
        with open(str(self.result_dir), mode, encoding="utf-8") as f:
            f.write(f"<s> {text} </s>\n")

    def multiprocessing(self):
        pool = Pool(cpu_cores-1)

        with tqdm.tqdm(total=self.total_files) as pbar:
            for _ in tqdm.tqdm(pool.imap_unordered(self.read, self.target_files)):
                pbar.update()

        pool.close()
        pool.join()

    def normal(self):
        for file_dir in self.target_files:
            self.read(file_dir)


    def __str__(self):
        return f"Total File: {self.total_files}"
    
    
######### 비출판물 전처리 ############
##국립국어원에서 제공해주는 데이터셋을 훈련을 위한 형태로 전처리
## 10754 files in 18s
class Malmunchi_book(Preprocess):
    def __init__(self):
        self.result_dir = Path("../../dataset/processed/modu_bee_corpus.txt")
        self.result_dir.parent.mkdir(exist_ok=True)

        self.target_files = glob.glob("../../dataset/NIKL_NP_v1.2_비출판물/국립국어원 비출판물 말뭉치(버전 1.2)/*.sjml")
        self.total_files = len(self.target_files)

    def read(self, file_dir):
        text = read_text_from_xml(file_dir)
        text = text.strip()
        if len(text) < 200:
            return 

        self.write(text)


######### 문어 말뭉치 전처리 ###########
#### 10045 Files in 36s
class Munu(Preprocess):
    def __init__(self):
        dataset_dir = Path("../../dataset/NIKL_WRITTEN(v1.2)")
        super().__init__(dataset_dir)

    def read(self, file_dir):
        with open(file_dir, "r", encoding="utf-8") as f:
            jsonString = json.load(f)
        paragraph = jsonString["document"][0]["paragraph"]
        text = " ".join([clean_text(data["form"]) for data in paragraph])
        text = text.strip()
        if len(text) < 200:
            return 

        self.write(text)


######## 029.대규모 구매도서 기반 한국어 말뭉치 데이터
#### 1022 Files in 01:22
class EnormousBookCorpus(Preprocess):
    def __init__(self):
        ## if zipped
        ## self.unzip("../../dataset/029.대규모 구매도서 기반 한국어 말뭉치 데이터/01.데이터/1.Training/라벨링데이터/TL_unscramble.zip")
        dataset_folder = Path("../../dataset/029.대규모 구매도서 기반 한국어 말뭉치 데이터/01.데이터/malmungchi/")
        self.result_dir = Path("../../dataset/processed/corpus.txt")
        self.result_dir.parent.mkdir(exist_ok=True)

        self.target_files = glob.glob(str(dataset_folder/"*"/"*"/"*.json"))
        self.total_files = len(self.target_files)

    def unzip(self, zip_dir):
        path_to_zip_file = Path(zip_dir)
        directory_to_extract_to = Path("../../dataset/029.대규모 구매도서 기반 한국어 말뭉치 데이터/01.데이터/malmungchi")
        directory_to_extract_to.mkdir(exist_ok=True)

        with zipfile.ZipFile(str(path_to_zip_file), 'r') as zip_ref:
            zip_ref.extractall(str(directory_to_extract_to))

    def read(self, file_dir):
        if "INFO" in file_dir: return
        with open(file_dir, "r", encoding="utf-8") as f:
            jsonString = json.load(f)
        paragraphs = jsonString["paragraphs"]

        text = ""
        for paragraph in paragraphs:
            sentences = paragraph["sentences"]
            text += " ".join([clean_text(data["text"]) for data in sentences])
        text = text.strip()
        if len(text) < 200:
            return 

        self.write(text)


####### 전문분야 말뭉치 #######
#### Total 5.78GB in 40s
class Expertise(Preprocess):
    def __init__(self):
        dataset_folder = Path("../../dataset/전문분야 말뭉치/corpus/training_논문")
        super().__init__(dataset_folder)

    def read(self, file_dir):
        with open(file_dir, "r", encoding="utf-8") as f:
            jsonString = json.load(f)
        datas = jsonString["data"]

        for data in datas:
            rows = data["rows"]
            text = " ".join([clean_text(row["text"]) for row in rows])
            text = text.strip()

            if len(text) < 200:
                continue 

            self.write(text)


####### 논문자료 요약 ########
#### 614MB in 22s
class PaperSumary(Preprocess):
    def __init__(self):
        dataset_folder = Path("../../dataset/논문자료 요약/corpus/training_논문")
        super().__init__(dataset_folder)

    def read(self, file_dir):
        with open(file_dir, "r", encoding="utf-8")  as f:
            jsonString = json.load(f)

        datas = jsonString["data"]

        for data in datas:
            summary_section = data["summary_section"][0]
            text = clean_text(summary_section["orginal_text"])
            text = text.strip()
            
            if len(text) < 200:
                continue

            self.write(text)


##### NAMUU ######
##### 565293 rows ######
class NAMU(Preprocess):
    def __init__(self):
        self.result_dir = Path("../../dataset/processed/namu_corpus.txt")
        self.result_dir.parent.mkdir(exist_ok=True)
        from datasets import load_dataset
        self.dataset = load_dataset("heegyu/namuwiki-extracted")["train"]
        self.total_files = len(self.dataset)

    def process_namu(self, data):
        text = data["text"]
        title = data["title"]

        text = clean_text(text)

        to_delete_prefix = ["width", "heigh"]

        splitted_text = text.split("\n")
        def isitin(t): 
            for prefix in to_delete_prefix:
                if prefix in t:
                    return False
            return True

        result = " ".join(filter(isitin, splitted_text))
        self.write(f"<s> {title}은 {result} </s>")

    def multiprocessing(self):
        pool = Pool(cpu_cores-1)

        with tqdm.tqdm(total=self.total_files) as pbar:
            for _ in tqdm.tqdm(pool.imap_unordered(self.process_namu, self.dataset)):
                pbar.update()

        pool.close()
        pool.join()

    def normal(self):
        for data in self.dataset:
            self.process_namu(data)


if __name__ == "__main__":
    process1 = Malmunchi_book()
    process1.multiprocessing()
    process2 = Munu()
    process2.multiprocessing()
    # process3 = EnormousBookCorpus()
    # process3.multiprocessing()
    # process4 = Expertise()
    # print(process4)
    # process4.multiprocessing()
    # process5 = PaperSumary()
    # process5.multiprocessing()
    # process6 = NAMU()
    # process6.multiprocessing()