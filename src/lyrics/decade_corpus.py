from os import listdir
from os.path import isfile, join

import requests
from nltk.corpus import stopwords

with open("data/misc/stoplist.txt", "r") as infile:
    sws = infile.read().split("\n")
#TODO: add refr, refrén etc to STOP_WORDS
STOP_WORDS = set(stopwords.words())
STOP_WORDS.add("refr")
STOP_WORDS.add("refrén")
for sw in sws:
    STOP_WORDS.add(sw.strip().lower())
data_path = "data/raw/decades"
out_path = "data/processed/decades"

good_pos = ["Adj", "V", "Adv", "N"]


def verbify(tag):
    if tag.startswith("V"):
        return "V"
    elif tag.startswith("Adj"):
        return "Adj"
    elif tag.startswith("Adv"):
        return "Adv"
    elif "_" in tag:
        return tag.split("_")[0]
    else:
        return tag


fs = [f for f in listdir(data_path) if isfile(join(data_path, f))]
decades_corpus = {}
for f in fs:
    with open(join(data_path, f), "r") as infile:
        c = infile.read()
    filtered_text = []
    for l in c.split("\n"):
        r = requests.post('http://127.0.0.1:5000/tok/morph/pos',
                          data={'text': l})
        raw_answer = r.text.split("\n")
        for ans in raw_answer:
            try:
                info = eval(ans.split("\t")[2])[0]
                lemma = info["lemma"].lower()
                pos = info["tag"].replace("/", "").replace("[", "").replace("]","").split("|")
                pos = [verbify(tag) for tag in pos][0]
                if pos in good_pos and lemma not in STOP_WORDS and len(
                        lemma) > 2:
                    filtered_text.append(lemma)
                    # print(lemma)
            except Exception as e:
                # print(ans, e)
                continue

    filtered_text = " ".join(filtered_text)
    decades_corpus[f] = filtered_text

# just a backup of the filtered and lemmatized texts
for k, v in decades_corpus.items():
    with open(join(out_path, k), "w") as outfile:
        outfile.write(v)
