"""
отбрасываем окончания слов


"""



import nltk
from nltk.corpus import stopwords
import re
import pymorphy2
from pymystem3 import Mystem
morph = pymorphy2.MorphAnalyzer()


def work_to_words(comments_array):
    sentence = []
    stopwords_ru = stopwords.words("russian")

    # знаки препинания, эмоджи и буквы
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    letters = '1234567890ЁЙЦУКЕНГШЩЗФЫВАПРОЛДЖЭЯЧСМИТЬБЮёйцукенгшщзхъфывапролджэячсмитьбю'
    lemmatizer = Mystem()

    for i in comments_array:
        #print('first', i)
        for j in i:
            # удаляем знаки пунктуации
            if j in punctuation:
                i = i.replace(j," ")
            # удаляем буквы
            if j not in letters:
                i = i.replace(j," ")
        i = i.lower().split()
        #print('second', i)
        text = ''
        for j in i:

            #j = emoji_pattern.sub(r'', j)   # удаляем эмоджи
            x = j
            # удаляем стоп слова


            if j in stopwords_ru:
                try:
                    i.remove(j)
                    x = j
                except:
                    print("error",j,i)
        i = lemmatizer.lemmatize(str(i))
        string = ''
        for j in i:
            if j.isalpha():
                string += j + ' '
        sentence.append(string)
        #print("type", string)
    return sentence






