from collections import Counter
from os import listdir
from os.path import isfile, join

import nltk
import networkx as nx
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
out_path = "data/processed/graphs"
freq_path = "data/processed/freqs"

lyrics_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
for lyrics_file in lyrics_files:
    with open(join(data_path, lyrics_file), "r") as infile:
        text = infile.read().strip().split()

    wfreqs = Counter(text)
    wfreqs = {k: v for k, v in sorted(wfreqs.items(),
                                      key=lambda item: item[1],
                                      reverse=True) if v > 2 and k not in STOP_WORDS}

    skipgrams = Counter(list(nltk.skipgrams(text, 2, 3)))
    skipgrams = {k: v for k, v in sorted(skipgrams.items(),
                                         key=lambda item: item[1],
                                         reverse=True) if v > 2}

    text_graph = nx.Graph()
    for k,v in skipgrams.items():
        text_graph.add_edge(k[0], k[1], weight=v)
    pr = nx.pagerank(text_graph, weight='weight')
    pr = {k: v for k, v in sorted(pr.items(),
                                  key=lambda item: item[1],
                                  reverse=True) if k not in STOP_WORDS}

    i = 0
    with open(join(out_path, lyrics_file[:-3] + "tsv"), "w") as outfile:
        for k,v in pr.items():
            if i < 200:
                o = k + "\t" + str(v) + "\n"
                outfile.write(o)
                i += 1

    j = 0
    with open(join(freq_path, lyrics_file[:-3] + "tsv"), "w") as outfile:
        for k,v in wfreqs.items():
            if j < 200:
                o = k + "\t" + str(v) + "\n"
                outfile.write(o)
                j += 1
