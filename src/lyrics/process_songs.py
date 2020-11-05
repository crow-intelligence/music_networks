import os

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
import termplotlib as tpl


Base = declarative_base()


# Schema
class Song(Base):
    __tablename__ = "song"
    id = Column(Integer, primary_key=True)
    lyrics = Column(UnicodeText, index=True)
    title = Column(String(1000), nullable=False, index=True, unique=True)
    # title contains the name of the performer, so it can be unique
    UniqueConstraint(lyrics, title, name="UniqueSong")


class Song2Year(Base):
    __tablename__ = "song2year"
    id = Column(Integer, primary_key=True)
    songid = Column(Integer, ForeignKey("song.id"))
    year = Column(Integer, nullable=False)
    UniqueConstraint(songid, year, name="UiqueS2y")


class Song2Title(Base):
    __tablename__ = "song2title"
    id = Column(Integer, primary_key=True)
    songid = Column(Integer, ForeignKey("song.id"))
    songtitle = Column(String(2000), nullable=False, index=True, unique=True)


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

Base.metadata.create_all(engine)

session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)
session = Session()

years = []
for value in session.query(Song2Year.year).distinct():
    years.append(value)

years = sorted(years)
years = [y[0] for y in years if y[0] > 0 and y[0] < 2021]


def year2decade(y):
    if len(str(y)) == 4:
        s = str(y)[:-1]
        s += "0"
        return int(s)
    else:
        return 0


decades = set([year2decade(y) for y in years])
decades_texts = {}
for d in decades:
    decades_texts[d] = []


session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)
session = Session()


def is_hun_line(line):
    try:
        line = line.split()
        line = " ".join([e.strip() for e in line])
        detected_langs = detect_langs(line)
        langs = [e.lang for e in detected_langs]
        probs = [e.prob for e in detected_langs]
        dp = dict(zip(langs, probs))
        if "hu" in dp and dp["hu"] > 0.8:
            return True
        else:
            return False
    except Exception as e:
        return False


for y in years:
    decade = year2decade(y)
    r = session.query(Song2Year).filter_by(year=y).all()
    r = [e.songid for e in r]
    for songid in r:
        q = session.query(Song).filter_by(id=songid).first()
        lyrics = q.lyrics.strip()
        lyrics = lyrics.replace("/", " ")
        lyrics = lyrics.replace(":|", " ")
        lyrics = lyrics.replace("//:", " ")
        lyrics = lyrics.replace("://", " ")
        lyrics = q.lyrics.split("\n")
        lyrics = [e.strip() for e in lyrics]
        lyrics = [e for e in lyrics if is_hun_line(e)]
        if lyrics:
            lyrics = " ".join(lyrics)
            lyrics = clean(lyrics, lower=False, fix_unicode=True, to_ascii=False, lang="hu")
            if lyrics != "Keressük a dalszöveget!" and len(lyrics ) > 40:
                decades_texts[decade].append(lyrics)

out_path = "data/raw/decades"
for k, v in decades_texts.items():
    fname = os.path.join(out_path, str(k) + ".txt")
    dcorpus = "\n".join(v)
    with open(fname, "w") as outfile:
        outfile.write(dcorpus)

desc_stat = zip(decades_texts.keys(), (len(v) for v in decades_texts.values()))
desc_stat = sorted(desc_stat, key=lambda t: t[0])
counts = [e[1] for e in desc_stat]
bins = [e[0] for e in desc_stat]

fig = tpl.figure()
fig.barh(counts, bins, force_ascii=False)
fig.show()
