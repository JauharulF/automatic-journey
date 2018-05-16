from wordcloud import WordCloud, STOPWORDS
import os
from PIL import Image
import numpy as np

currdir = os.path.dirname(__file__)
j_stop = {'iki', 'iku', 'aku', 'ku', 'ak', 'aq', 'tak', 'kok', 'sing', 'yo', 'ae', 'wis', 'wes',
          'gak', 'lek', 'lho', 'ta', 'di', 'ya', 'nang', 'opo', 'nek', 'karo', 'sek',
          'tah', 'lha', 'sik', 'itu', 'loh', 'tok', 'kita', 'lah', 'ben', 'ke', 'tp', 'ga', 'yg',
          'awakmu', 'ono', 'kan', 'rek', 'iso', 'mu'}

def create_wordcloud(text):
    mask = np.array(Image.open(os.path.join(currdir, "cloud.png")))
    sword= set(STOPWORDS)
    sword.update(j_stop)
    # print(sword)
    wc = WordCloud(background_color="lightcyan", width=640, height=480, max_words=200, stopwords=sword)
    wc.generate(text)
    wc.to_file(os.path.join(currdir, "wc.png"))

with open('spanda98_.txt', encoding='utf8') as fin:
    teks = ""
    for line in fin:
        teks += line
    create_wordcloud(teks)
