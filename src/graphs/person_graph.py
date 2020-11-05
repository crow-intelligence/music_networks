import os
import itertools
from collections import Counter

import networkx as nx
from cleantext import clean
from langdetect import detect_langs
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    func,
    ForeignKey,
    String,
    UnicodeText,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

# db type, dialect, auth, location, port, dbname
user = "root"
password = "secret"
# host = "127.0.0.1"
host = "localhost"
port = 3306
dbname = "music_hu"
db = f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"
# create engine
engine = create_engine(
    db,
    pool_recycle=10,
    echo=False,
    pool_size=300,
    max_overflow=20,
    encoding="utf-8",
    convert_unicode=True,
    connect_args={"connect_timeout": 150},
)

Base = declarative_base()


class Performer(Base):
    __tablename__ = "performer"
    id = Column(Integer, primary_key=True)
    name = Column(String(400), nullable=False, index=True)
    aka = Column(String(400), index=True)
    UniqueConstraint(name, aka, name="N2A")


class Person(Base):
    __tablename__ = "person"
    id = Column(Integer, primary_key=True)
    name = Column(String(400), nullable=False, index=True)
    uri = Column(String(400), index=True)
    UniqueConstraint(name, uri, name="UniquePerson")


class Person2Performer(Base):
    __tablename__ = "person2performer"
    id = Column(Integer, primary_key=True)
    performerid = Column(Integer, ForeignKey("performer.id"))
    personid = Column(Integer, ForeignKey("person.id"))
    UniqueConstraint(performerid, personid, name="UniqueP2P")


Base.metadata.create_all(engine)

session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)
session = Session()

performer_ids = []
for value in session.query(Performer.id).distinct():
    performer_ids.append(value[0])

G = nx.Graph()
edges = []
for performer_id in performer_ids:
    q = session.query(Person2Performer).filter_by(
        performerid=performer_id).all()
    persons = [p.personid for p in q]
    uris = []
    pnames = []
    for p in persons:
        pq = session.query(Person).filter_by(id=p).first()
        pname = pq.name
        puri = "http://zeneszoveg.hu/" + pq.uri
        uris.append(puri)
        pnames.append(pname)
        G.add_node(puri, label=pname)
    connections = list(itertools.combinations(uris, 2))
    edges.extend(connections)


edges = Counter(edges)
for k,v in edges.items():
    G.add_edge(k[0], k[1], weight=v)

nx.write_graphml(G, "data/persons_full.graphml")


toberemoved = []
for e in G.degree:
    if e[1] < 2:
        toberemoved.append(e[0])

for k in edges:
    w = edges[k]
    if k[0] in toberemoved or k[1] in toberemoved or w < 2:
        if (k[0], k[1]) in G.edges:
            G.remove_edge(k[0], k[1])

for e in toberemoved:
    G.remove_node(e)

nx.write_graphml(G, "data/persons_filtered.graphml")

print(len(G.nodes), len(G.edges))
