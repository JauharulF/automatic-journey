import MySQLdb
import MySQLdb.cursors
import tensorflow as tf
import numpy as np

from tensorflow import keras
from nltk.tokenize import word_tokenize

def create_embedding(finput=None, foutput=None):
    word_dict = {}
    with open(finput, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            for word in word_tokenize(line):
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
    with open(foutput, 'w', encoding='utf-8') as fout:
        for word in sorted(word_dict, key=word_dict.get, reverse=True):
            fout.write(word + ';'+ str(word_dict[word]) + "\n")

# create embedding from csv file resulted
def get_embedding(fname=None, num_words=None):
    word_dict = {}
    with open(fname, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            word = line.split(';')[0]
            if num_words and num_words>i:
                word_dict[word] = i
            elif num_words <= i:
                break
    # done reading
    word_index = {k:(v+4) for k,v in word_dict.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    # done
    return word_index

def get_dataset(conn=None, datatype=None, word_index=None):
    cursor = conn.cursor()
    t_data = []
    t_label= []
    sql = """ SELECT id, en_data, ms_data, label FROM sentiment.imdb_{} ORDER BY id """.format(datatype)
    cursor.execute(sql)
    rs = cursor.fetchall()
    for row in rs:
        for src in ['en_data', 'ms_data']:
            datum = []
            for word in row['en_data'].split():
                if word in word_index:
                    datum.append(word_index[word])
                else:
                    datum.append(2)
            t_data.append(datum)
        t_label.append(row['label'])
        t_label.append(row['label'])
    return t_data, t_label

def load_data(conn=None, word_index=None):
    train_data, train_labels = get_dataset(conn=conn, datatype="train", word_index=word_index)
    test_data, test_labels = get_dataset(conn=conn, datatype="test", word_index=word_index)
    return (train_data, train_labels), (test_data, test_labels)

def predict(model=None, sentence=None, word_index=None):
    sent_arr = []
    for word in word_tokenize(sentence.lower()):
        sent_arr.append(word_index[word] if word in word_index else 2) # 2 is <UNK>
    sent_pad = keras.preprocessing.sequence.pad_sequences([sent_arr], value=word_index["<PAD>"], padding='post', maxlen=256)
    return model.predict(sent_pad)

def update_polarity(conn=None, word_index=None, model=None):
    cursor = conn.cursor()
    sql = """ SELECT id, sentence FROM sentiment.socmed ORDER BY id """
    cursor.execute(sql)
    rs = cursor.fetchall()
    for row in rs: 
        polar = predict(model=model, sentence=row['sentence'], word_index=word_index)
        sql = """ UPDATE sentiment.socmed SET polarity=%s WHERE id=%s """
        cursor.execute(sql, (polar[0][0], row['id']))
        conn.commit()

def predict_lang(sentence=None, word_index=None):
    count = 0
    for word in word_tokenize(sentence):
        if word in word_index:
            count += 1
    return (count / len(word_tokenize(sentence)))

def update_lang_prediction(conn=None, word_index=None, lang=None):
    cursor = conn.cursor()
    sql = """ SELECT id, sentence FROM sentiment.socmed ORDER BY id """
    cursor.execute(sql)
    rs = cursor.fetchall()
    for row in rs: 
        predict = predict_lang(sentence=row['sentence'], word_index=word_index)
        sql = """ UPDATE sentiment.socmed SET {}_prediction=%s WHERE id=%s """.format(lang)
        cursor.execute(sql, (predict, row['id']))
        conn.commit()

word_index = get_embedding(fname="wordindex.en-my.txt", num_words=29996)
dbcon = MySQLdb.connect(host='localhost', user='jauharul', passwd='123456', db='sentiment', cursorclass=MySQLdb.cursors.DictCursor)
(train_data, train_labels), (test_data, test_labels) = load_data(conn=dbcon, word_index=word_index)

# try to run the keras 
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=0, padding='post', maxlen=256)
test_data  = keras.preprocessing.sequence.pad_sequences(test_data,  value=0, padding='post', maxlen=256)
# the vocab size will be 10000 + 4 special chars
vocab_size = 30000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Convolution1D(nb_filter=80, filter_length=4, border_mode='valid', activation='relu')) # dikomen soalnya masih gagal
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])
# split 20000 data for validation ~ 40%
x_val = train_data[:20000]
partial_x_train = train_data[20000:]
y_val = train_labels[:20000]
partial_y_train = train_labels[20000:]
history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
results = model.evaluate(test_data, test_labels)
print(results)

update_polarity(conn=dbcon, word_index=word_index, model=model)
update_lang_prediction(conn=dbcon, word_index=word_index, lang="en")
update_lang_prediction(conn=dbcon, word_index=word_index, lang="ms")
