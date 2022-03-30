from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

tokenizer = RegexTokenizer()
model = FastTextSocialNetworkModel(tokenizer=tokenizer)

messages = [
    'Сегодня хорошая погода',
    'Я счастлив проводить с тобою время',
    'Мне нравится эта музыкальная композиция',
    'В больнице была ужасная очередь',
    'Сосед с верхнего этажа мешает спать',
    'Маленькая девочка потерялась в торговом центре',
]

results = model.predict(messages, k=2)
for message, sentiment in zip(messages, results):
    print(message, '-&gt;', sentiment)