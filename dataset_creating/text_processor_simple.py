import os
import subprocess
import re
import fastText
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class TextProcessor:
    def __init__(self):
        """
        Now for english language only
        """
        self._language_model = fastText.load_model("models/lid.176.bin")
        return

    def preprocess(self, text: str):
        lowercase = text.lower()
        text_cleaned = self.clean_text(lowercase)

        if self.__detect_language__(text_cleaned) != "en":
            return ""

        tokens = self.__europalp_tokenizer__(text_cleaned, "en")
        tokens = self.drop_stopwords(tokens)

        ps = PorterStemmer()
        stemmed = [ps.stem(word) for word in tokens]
        print(stemmed)
        return ' '.join(stemmed)

    @staticmethod
    def clean_text(text: str):
        link_regexp = 'https?:[^\s]+'
        uname_regexp = '@[^\s]+'
        res = re.sub(link_regexp, '', text)
        res = re.sub(uname_regexp, '', res)
        for c in ['.', ',', ':', ';', '"', '?', '!', '—', '-', '/']:
            res = res.replace(c, ' ')
        return res

    def __detect_language__(self, text: str):
        predicted = self._language_model.predict(text)[0][0]
        return predicted.replace("__label__", "")

    @staticmethod
    def drop_stopwords(words: list):
        return list(filter(
            lambda x: x not in stopwords.words('english') and x.isalpha(),
            words))

    @staticmethod
    def __europalp_tokenizer__(text: str, lang: str):
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(
            ["perl", "tools/europalp_tokenizer/tokenizer.perl", "-l", lang],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=FNULL
        )
        output = p.communicate(input=text.encode())[0]
        tokens = output.decode().replace("\n", "").split(" ")

        return tokens


if __name__ == "__main__":
    # simple test
    txt_en = "MT@xeni All you need to know abt why there r there riots in" \
             " Baltimore is in these charts. http://t.co/2xdobenYtf " \
             "http://t.co/EeoXqLGJ2E #fb"

    tp = TextProcessor()
    print(tp.preprocess(txt_en))