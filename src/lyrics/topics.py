from os import listdir
from os.path import isfile, join

from gensim.corpora import Dictionary
from gensim.models import LdaMulticore

stop_extension = ['érez', 'tesz', 'áll', 'régi', 'fáj', 'nap', 'hív', 'rég',
                  'játszik', 'gondol', "jobb", "rossz", "talál", "vesz",
                  "kicsi", "szabad", "mér", "visz", 'keres', 'kap', 'boldog',
                  "álmodik", 'messze', 'elmegy', 'mély', 'figyel', 'igaz', 'vigyáz', 'lassú', 'ért', 'eszik', 'csinál']

with open("data/misc/stoplist.txt", "r") as infile:
    stoplist = infile.read().split("\n")
    stoplist.extend(stop_extension)
    stoplist = set(stoplist)

in_path = "data/processed/decades"
corpus_files = [f for f in listdir(in_path) if isfile(join(in_path, f))]
docs = []
for cf in corpus_files:
    with open(join(in_path, cf), "r") as infile:
        txt = infile.read().split()
        txt = [wd for wd in txt if wd not in stoplist]
        docs.append(txt)

dictionary = Dictionary(docs)
id2word = dictionary.id2token

corpus = [dictionary.doc2bow(doc) for doc in docs]

lda_model = LdaMulticore(corpus=corpus,
                         id2word=dictionary,
                         random_state=100,
                         num_topics=5,
                         passes=10,
                         chunksize=1000,
                         batch=False,
                         alpha='asymmetric',
                         decay=0.5,
                         offset=64,
                         eta=None,
                         eval_every=0,
                         iterations=100,
                         gamma_threshold=0.001,
                         per_word_topics=True)

lda_model.save('models/lda_model.model')

for i in range(5):
    wds = lda_model.get_topic_terms(i, 10)
    wds = [e[0] for e in wds]
    wds = [dictionary.id2token[e] for e in wds]
    print(i, wds)
