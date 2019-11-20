import operator
from collections import Counter
from itertools import combinations

import networkx as nx
from sklearn.preprocessing import minmax_scale
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, UnicodeText
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

###############################################################################
#####                               DB                                    #####
###############################################################################
db = "sqlite:///teszt.db"
Base = declarative_base()


# Schema
class Song(Base):
    __tablename__ = "song"
    id = Column(Integer, primary_key=True)
    lyrics = Column(UnicodeText)
    title = Column(String(1000))


class Song2Year(Base):
    __tablename__ = "song2year"
    id = Column(Integer, primary_key=True)
    songid = Column(Integer, ForeignKey("song.id"))
    year = Column(Integer)


class Song2Title(Base):
    __tablename__ = "song2title"
    id = Column(Integer, primary_key=True)
    songid = Column(Integer, ForeignKey("song.id"))
    songtitle = Column(String(2000))


class Composer(Base):
    __tablename__ = "composer"
    id = Column(Integer, primary_key=True)
    name = Column(String(400))
    aka = Column(String(400))


class Author(Base):
    __tablename__ = "author"
    id = Column(Integer, primary_key=True)
    name = Column(String(400))
    aka = Column(String(400))


class Performer(Base):
    __tablename__ = "performer"
    id = Column(Integer, primary_key=True)
    name = Column(String(400))
    aka = Column(String(400))


class Person(Base):
    __tablename__ = "person"
    id = Column(Integer, primary_key=True)
    name = Column(String(400))
    uri = Column(String(400))


class Person2Performer(Base):
    __tablename__ = "person2performer"
    id = Column(Integer, primary_key=True)
    performerid = Column(Integer, ForeignKey("performer.id"))
    personid = Column(Integer, ForeignKey("person.id"))


class Performer2Song(Base):
    __tablename__ = "performer2song"
    id = Column(Integer, primary_key=True)
    songid = Column(Integer, ForeignKey("song.id"))
    performerid = Column(Integer, ForeignKey("performer.id"))


class Author2Song(Base):
    __tablename__ = "author2song"
    id = Column(Integer, primary_key=True)
    songid = Column(Integer, ForeignKey("song.id"))
    authorid = Column(Integer, ForeignKey("author.id"))


class Composer2Song(Base):
    __tablename__ = "composer2song"
    id = Column(Integer, primary_key=True)
    songid = Column(Integer, ForeignKey("song.id"))
    composerid = Column(Integer, ForeignKey("composer.id"))


# create engine
engine = create_engine(db, echo=False)
Base.metadata.create_all(engine)

session_factory = sessionmaker(bind=engine)
    Session = scoped_session(session_factory)
    session = Session()
###############################################################################
#####                              Dedupe                                 #####
###############################################################################
song_id = {}
id_song = {}
for row in session.query(Song2Title):
    songid = row.songid
    songtitle = row.songtitle
    if songtitle not in song_id and "Keressük " not in songtitle:
        song_id[songtitle] = songid
        id_song[songid] = songtitle

author_id = {}
id_author = {}
for row in session.query(Author):
    id = row.id
    name = row.name
    name = row.name.strip().title()
    if name not in author_id and "Keressük " not in name:
        author_id[name] = id
        id_author[id] = name

composer_id = {}
id_composer = {}
for row in session.query(Composer):
    id = row.id
    name = row.name
    name = row.name.strip().title()
    if name not in composer_id and "Keressük " not in name:
        composer_id[name] = id
        id_composer[id] = name

performer_id = {}
id_performer = {}
for row in session.query(Performer):
    id = row.id
    name = row.name
    name = row.name.strip().title()
    if name not in performer_id and "Keressük " not in name:
        performer_id[name] = id
        id_performer[id] = name

person_id = {}
id_person = {}
person_uri = {}
for row in session.query(Person):
    id = row.id
    name = row.name
    name = row.name.strip().title()
    uri = row.uri
    if name not in person_id and "Keressük " not in name:
        person_id[name] = id
        id_person[id] = name
        person_uri[name] = uri

###############################################################################
#####                       Person graphs                                 #####
###############################################################################
author2song = []
for row in session.query(Author2Song):
    songid = row.songid
    authorid = row.authorid
    if songid in id_song and authorid in id_author:
        t = (songid, authorid)
        author2song.append(t)

composer2song = []
for row in session.query(Composer2Song):
    songid = row.songid
    composerid = row.composerid
    if songid in id_song and composerid in id_composer:
        t = (songid, composerid)
        composer2song.append(t)

performer2song = []
for row in session.query(Performer2Song):
    songid = row.songid
    performerid = row.id
    if songid in id_song and performerid in id_performer:
        t = (songid, performerid)
        performer2song.append(t)

performer2person = []
for row in session.query(Person2Performer):
    preformerid = row.performerid
    personid = row.personid
    if performerid in id_performer and personid in id_person:
        t = (preformerid, personid)
        performer2person.append(t)

ppl_nodes = set()
ppl_edges = []
for id in id_song:
    author_ids = [e[1] for e in author2song if e[0] == id]
    authors = [id_author[e] for e in author_ids]

    composer_ids = [e[1] for e in composer2song if e[0] == id]
    composers = [id_composer[e] for e in composer_ids]

    performer_ids = [e[1] for e in performer2song if e[0] == id]
    person_ids = [e[1] for e in performer2person if e[0] in performer_ids]
    persons = [id_person[e] for e in person_ids]

    all_persons = set(authors + composers + persons)
    ppl_nodes.update(all_persons)
    connections = combinations(all_persons, 2)
    for con in connections:
        t = tuple(sorted(con))
        ppl_edges.append(t)

G = nx.Graph()
for node in ppl_nodes:
    G.add_node(node)

ppl_edges_weighted = Counter(ppl_edges)
for k, v in ppl_edges_weighted.items():
    G.add_edge(k[0], k[1], weight=v)

pr = nx.pagerank(G)
rnodes = list(pr.keys())
rescaled_pr = list(pr.values())
rescaled_pr = list(minmax_scale(rescaled_pr, [1, 5]))
pr_rescaled = dict(zip(rnodes, rescaled_pr))
sorted_pr_rescaled = sorted(pr_rescaled.items(),
                            key=operator.itemgetter(1),
                            reverse=True)
sorted_pr_rescaled = [e[0] for e in sorted_pr_rescaled][:400]
tobedeleted = [e for e in rnodes if e not in sorted_pr_rescaled]
for node in tobedeleted:
    G.remove_node(node)

giant = list(max(nx.connected_components(G), key=len))

with open("data/nodes.tsv", "w") as f:
    for node in giant:
        o = node + "\t" + str(pr_rescaled[node]) + "\n"
        f.write(o)

ppl_edge_weights = list(ppl_edges_weighted.values())
ppl_edges_weights_rescaled = list(minmax_scale(ppl_edge_weights, [1,5]))
ppl_edges = list(ppl_edges_weighted.keys())
ppl_edges_weighted_rescaled = dict(zip(ppl_edges, ppl_edges_weights_rescaled))

with open("data/edges.tsv", "w") as f:
    for k,v in ppl_edges_weighted_rescaled.items():
        if k[0] in giant and k[1] in giant:
            o = k[0] + "\t" + k[1] + "\t" + str(v) + "\n"
            f.write(o)

nx.write_graphml(G, "data/ppl.graphml")

###############################################################################
#####                         Song graphs                                 #####
###############################################################################
# song_nodes = set()
# song_edges = []
# for id in id_song:
#     song_title = id_song[id].split("|")[0]
#     song_authors = [e[1] for e in author2song if e[0] == id]
#     authors_songs = [e[0] for e in author2song if e[1] in song_authors]
#     if authors_songs:
#         authors_titles = [id_song[e].split("|")[0] for e in authors_songs]
#     else:
#         authors_titles = []
#
#     song_composers = [e[1] for e in composer2song if e[0] == id]
#     composers_songs = [e[0] for e in composer2song if e[1] in song_composers]
#     if composers_songs:
#         composers_titles = [id_song[e].split("|")[0] for e in composers_songs]
#     else:
#         composers_titles = []
#
#     song_performers = [e[1] for e in performer2song if e[0] == id]
#     performers_songs = [e[0] for e in performer2song if e[1] in song_performers]
#     if performers_songs:
#         performers_titles = [id_song[e].split("|")[0] for e in performers_songs]
#     else:
#         performers_titles = []
#
#     all_contributors = authors_titles + composers_titles + performers_titles
#     all_contributors = set(all_contributors)
#     song_nodes.update(all_contributors)
#     song_nodes.add(song_title)
#     for title in sorted(all_contributors):
#         t = tuple(sorted([song_title, title]))
#         song_edges.append(t)
#
# song_edges_weighted = Counter(song_edges)
# sorted_edges = sorted(song_edges_weighted.items(), key=operator.itemgetter(1), reverse=True)
#
# G2 = nx.Graph()
#
# for e in sorted_edges:
#     nodes, weight = e[0], e[1]
#     n1, n2 = nodes[0], nodes[1]
#     if n1 != n2:
#         G2.add_edge(n1, n2, weight=weight)
#         G2.add_node(n1)
#         G2.add_node(n2)
#
# G3 = nx.Graph()
# filtered_nodes = set([e[0] for e in list(G2.degree) if e[1] > 300])
# filtered_edges = [e[0] for e in sorted_edges if e[0][0] in filtered_nodes and e[0][1] in filtered_nodes]
# print(len(filtered_edges))
#
# for e in filtered_edges:
#     n1, n2 = e[0], e[1]
#     t = sorted([n1, n2])
#     G3.add_node(n1)
#     G3.add_node(n2)
#     G3.add_edge(t[0], t[1])
#
# nx.write_graphml(G3, "data/songs.graphml")
