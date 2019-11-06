from collections import Counter
from itertools import combinations

import networkx as nx
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
    if name not in author_id and "Keressük " not in name:
        author_id[name] = id
        id_author[id] = name

composer_id = {}
id_composer = {}
for row in session.query(Composer):
    id = row.id
    name = row.name
    if name not in composer_id and "Keressük " not in name:
        composer_id[name] = id
        id_composer[id] = name

performer_id = {}
id_performer = {}
for row in session.query(Performer):
    id = row.id
    name = row.name
    if name not in performer_id and "Keressük " not in name:
        performer_id[name] = id
        id_performer[id] = name

person_id = {}
id_person = {}
person_uri = {}
for row in session.query(Person):
    id = row.id
    name = row.name
    uri = row.uri
    if name not in person_id and "Keressük " not in name:
        person_id[name] = id
        id_person[id] = name
        person_uri[name] = uri

###############################################################################
#####                         Song graphs                                 #####
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

ppl_edges_wieghted = Counter(ppl_edges)
for k, v in ppl_edges_wieghted.items():
    G.add_edge(k[0], k[1], weight=v)

nx.write_graphml(G, "data/ppl.graphml")
# 3 link persons to performers, authors and composers
# 4 generate a graph which links ppl through songs
# 5 make a language model for songs
# 6 generate a graph which links songs through ppl