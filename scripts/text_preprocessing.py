import re
import regex
from dataclasses import dataclass

@dataclass
class TextPreprocessing():
    # Replace unneeded Data with mapping marks.
    @staticmethod
    def replace(text):
        replace_mark = {'\\n': "\n", "\\'": "'", "‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": "sqrt", "×": "x", "²": "2",
                 "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e",
                 '∞': '무한', 'θ': '세타', '÷': '/', 'α': '알파', '•': '.', 'à': 'a', '−': '-', 'β': '베타',
                 '∅': '', '³': '3', 'π': '파이', '\u200b': '', '…': '.', '\ufeff': '', 'करना': '', 'है': ''}

        for mark in replace_mark:
            text = text.replace(mark, replace_mark[mark])

        return text

    @staticmethod
    def sub(text):
         # 이메일 제거
        email_pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'  
        text = re.sub(email_pattern, "", text)

        # 링크 제거
        link_pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        text = re.sub(link_pattern, "", text)

        # 일본어 한자 제거
        japanese_pattern = r"[\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Han}]"
        text = regex.sub(japanese_pattern, "", text)

        # 특수 문자 제거
        special_word_pattern = '[#/\:^$@*※~&%ㆍ』\\‘|\(\)\[\]\<\>`…》]'
        text = re.sub(special_word_pattern, "", text)

        # 줄바꿈 문자 제거
        text = re.sub('\n', '', text)

        # 연속된 공백 제거
        text = re.sub(r"\s+", " ", text)

        return text

    @staticmethod
    def preprocess(text):
        text = TextPreprocessing.replace(text)
        text = TextPreprocessing.sub(text)

        return text
