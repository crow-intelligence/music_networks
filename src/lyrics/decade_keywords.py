from collections import Counter
from os import listdir
from os.path import isfile, join

from corpus_toolkit import corpus_tools as ct
from nltk.corpus import stopwords


with open("data/misc/stoplist.txt", "r") as infile:
    sws = infile.read().split("\n")
#TODO: add refr, refrén etc to STOP_WORDS
STOP_WORDS = set(stopwords.words())
STOP_WORDS.add("refr")
STOP_WORDS.add("refrén")
for sw in sws:
    STOP_WORDS.add(sw.strip().lower())


data_path = "data/processed/decades"
out_path = "data/processed/keywords"

full_corpus = {}
fs = [f for f in listdir(data_path) if isfile(join(data_path, f))]
for f in fs:
    with open(join(data_path, f), "r") as infile:
        cp = infile.read().strip().split()
        cp = Counter(cp)
        for k,v in cp.items():
            k = k.strip().replace("-", "#")
            if k not in STOP_WORDS:
                if k in full_corpus:
                    full_corpus[k] += v
                else:
                    full_corpus[k] = v

for f in fs:
    with open(join(data_path, f), "r") as infile:
        cp = infile.read().strip().split()
        cp = [wd.replace("-", "#") for wd in cp if wd not in STOP_WORDS]
        cp = Counter(cp)
        key = ct.keyness(cp, full_corpus, effect="log-ratio")
        key = {
            k: v for k, v in sorted(key.items(), key=lambda item: item[1], reverse=True)
        }
        with open(join(out_path, f[:-3] + "tsv"), "w") as outfile:
            # h = "word\tll\n"
            # outfile.write(h)
            for k, v in key.items():
                o = k + "\t" + str(v) + "\n"
                outfile.write(o)
