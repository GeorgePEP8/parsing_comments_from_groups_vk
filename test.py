import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
import re

from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tflearn.data_utils import to_categorical
from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import TweetTokenizer



# Constants
#-----------------------------------------------------------------------#


POSITIVE_TWEETS_CSV = "positive.csv"
NEGATIVE_TWEETS_CSV = 'negative.csv'


# размер словаря
VOCAB_SIZE = 5000

#-----------------------------------------------------------------------#



# load data
#-----------------------------------------------------------------------#
# грузим текстовые данные в озу

tweets_col_number = 3

negative_tweets = pd.read_csv(
    'negative.csv', header=None, delimiter=';')[[tweets_col_number]]
positive_tweets = pd.read_csv(
    'positive.csv', header=None, delimiter=';')[[tweets_col_number]]

#-----------------------------------------------------------------------#




# stemmer
#-----------------------------------------------------------------------#

# для каждого слова выдаем базовую форму

stemer = RussianStemmer()

# отсеиваем все не кириллическое
regex = re.compile('[^а-яА-Я ]')
stem_cache = {}

def get_stem(token):
    stem = stem_cache.get(token, None)
    if stem:
        return stem
    token = regex.sub('', token).lower()
    stem = stemer.stem(token)
    stem_cache[token] = stem
    return stem

#-----------------------------------------------------------------------#



# Vocabulary creation
#-----------------------------------------------------------------------#
#создаем словарь с уникальными словами в позитивных и негативных твитах
stem_count = Counter()
tokenizer = TweetTokenizer()

def count_unique_tokens_in_tweets(tweets):
    for _, tweet_series in tweets.iterrows():
        tweet = tweet_series[3]
        tokens = tokenizer.tokenize(tweet)
        for token in tokens:
            stem = get_stem(token)
            stem_count[stem] += 1

count_unique_tokens_in_tweets(negative_tweets)
count_unique_tokens_in_tweets(positive_tweets)


#-----------------------------------------------------------------------#



# кол-во уникальных стемов
#-----------------------------------------------------------------------#
print("Total unique stems found: ", len(stem_count))

#-----------------------------------------------------------------------#



#-----------------------------------------------------------------------#
vocab = sorted(stem_count, key=stem_count.get, reverse=True)[:VOCAB_SIZE]
print(vocab[:100])
#-----------------------------------------------------------------------#



#-----------------------------------------------------------------------#
idx = 2
print("stem: {}, count: {}"
      .format(vocab[idx], stem_count.get(vocab[idx])))

#-----------------------------------------------------------------------#



#-----------------------------------------------------------------------#
token_2_idx = {vocab[i] : i for i in range(VOCAB_SIZE)}
len(token_2_idx)
#-----------------------------------------------------------------------#



#-----------------------------------------------------------------------#
#print(token_2_idx['сказа'])
#-----------------------------------------------------------------------#



#-----------------------------------------------------------------------#
def tweet_to_vector(tweet, show_unknowns=False):
    vector = np.zeros(VOCAB_SIZE, dtype=np.int_)
    for token in tokenizer.tokenize(tweet):
        stem = get_stem(token)
        idx = token_2_idx.get(stem, None)
        if idx is not None:
            vector[idx] = 1
        elif show_unknowns:
            print("Unknown token: {}".format(token))
    return vector


tweet = negative_tweets.iloc[1][3]
print("tweet: {}".format(tweet))
print("vector: {}".format(tweet_to_vector(tweet)[:10]))
print(vocab[5])

tweet_vectors = np.zeros(
    (len(negative_tweets) + len(positive_tweets), VOCAB_SIZE),
    dtype=np.int_)
tweets = []
for ii, (_, tweet) in enumerate(negative_tweets.iterrows()):
    tweets.append(tweet[3])
    tweet_vectors[ii] = tweet_to_vector(tweet[3])
for ii, (_, tweet) in enumerate(positive_tweets.iterrows()):
    tweets.append(tweet[3])
    tweet_vectors[ii + len(negative_tweets)] = tweet_to_vector(tweet[3])

labels = np.append(
    np.zeros(len(negative_tweets), dtype=np.int_),
    np.ones(len(positive_tweets), dtype=np.int_))


X = tweet_vectors
y = to_categorical(labels, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(y_test[:10])


# learning_rate=0.1 - скорость обучения нейросети
def build_model(learning_rate=0.1):
    tf.compat.v1.reset_default_graph()

    # создаем входной слой VOCAB_SIZE - кол-во нейронов
    net = tflearn.input_data([None, VOCAB_SIZE])
    # уровень с 125 нейронами
    net = tflearn.fully_connected(net, 125, activation='ReLU')
    # уровень с 25 нейронами
    net = tflearn.fully_connected(net, 25, activation='ReLU')
    # уровень с 2 нейронами
    net = tflearn.fully_connected(net, 2, activation='softmax')
    # алгоритм обучения нейронов
    regression = tflearn.regression(
        net,
        optimizer='sgd',
        learning_rate=learning_rate,
        loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    return model

# само обучение
model = build_model(learning_rate=0.15)

model.fit(
    X_train,
    y_train,
    validation_set=0.1,
    show_metric=True,
    batch_size=128,
    n_epoch=30) # кол-во эпох

# iter - кол-во полных циклов обучения
# acc - точность



model.save('location')
print('save')
model_loaded = keras.models.load_model('location')


#testing
#-----------------------------------------------------------------------#

predictions = (np.array(model.predict(X_test))[:, 0] >= 0.5).astype(np.int_)
accuracy = np.mean(predictions == y_test[:, 0], axis=0)
print("Accuracy: ", accuracy)


def test_tweet(tweet):
    # преобразовываем в вектор
    tweet_vector = tweet_to_vector(tweet, True)
    positive_prob = model_loaded.predict([tweet_vector])[0][1]
    print('Original tweet: {}'.format(tweet))
    print('P(positive) = {:.5f}. Result: '.format(positive_prob),
          'Positive' if positive_prob > 0.5 else 'Negative')


def test_tweet_number(idx):
    test_tweet(tweets[idx])


test_tweet_number(120705)




tweets_for_testing = ["Я сломал этот телефон",
"все закончилось плохо", " я заболел", " я приболел"]
for tweet in tweets_for_testing:
    test_tweet(tweet)
    print("---------")