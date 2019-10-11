import re

import requests
import bs4
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, UnicodeText
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

from src.data_tasks.requests_with_proxies import request_with_proxy

###############################################################################
#####                                Setup                                #####
###############################################################################
ua = UserAgent()

prefix = "http://www.zeneszoveg.hu/"
base_url = "http://www.zeneszoveg.hu/eloadok/"
initials = [
    "09",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]
stoplist = ["egyuttes/uj.html", "szemely/uj.html"]
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
    title = Column(String(500))


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

###############################################################################
#####                        Collect data                                 #####
###############################################################################
for ch in initials:
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX", ch)
    session_factory = sessionmaker(bind=engine)
    Session = scoped_session(session_factory)
    session = Session()

    url = base_url + ch
    header = {"User-Agent": ua.random}
    try:
        html = request_with_proxy(url)
        # html = requests.get(url, headers=header).text
        soup = BeautifulSoup(html, "lxml")
        links = soup.find_all("a")
        for link in links:
            if "href" in link.attrs:
                if link["href"].startswith("egyuttes/"):
                    if link["href"] not in stoplist:
                        band_name = link.text
                        band_page = prefix + link["href"]
                        header = {"User-Agent": ua.random}
                        try:
                            band_html = request_with_proxy(band_page)
                            # band_html = requests.get(band_page, headers=header).text
                            band_soup = BeautifulSoup(band_html, "lxml")
                            band_links = band_soup.find_all("a")
                            members = [
                                l
                                for l in band_links
                                if "href" in l.attrs
                                and l["href"].startswith("szemely/")
                                and l["href"] not in stoplist
                            ]
                            member_id = {}
                            members2add = []
                            for l in members:
                                member_id[l.text] = l["href"]
                            for member, uri in member_id.items():
                                member_query = (
                                    session.query(Person).filter_by(name=member).first()
                                )
                                if not member_query:
                                    member_entry = Person(name=member, uri=uri)
                                    session.add(member_entry)
                                    session.flush()
                                    session.refresh(member_entry)
                                    session.commit()
                                    memberid = member_entry.id
                                else:
                                    memberid = member_query.id
                                members2add.append(memberid)
                            for band_link in band_links:
                                if band_link["href"].startswith("dalszoveg/"):
                                    if "href" in band_link.attrs:
                                        song_title = band_link.text
                                        header = {"User-Agent": ua.random}
                                        song_url = prefix + band_link["href"]
                                        try:
                                            song_html = request_with_proxy(song_url)
                                            # song_html = requests.get(
                                            #     song_url, headers=header
                                            # ).text
                                            song_soup = BeautifulSoup(song_html, "lxml")
                                            song_text = song_soup.find_all(
                                                "div", class_="lyrics-plain-text"
                                            )
                                            song_text = [s.text for s in song_text][0]
                                            song_infos = song_soup.find_all(
                                                "div",
                                                {"class": "lyrics-header-text short"},
                                            )
                                            song_entry = Song(
                                                title=song_title, lyrics=song_text
                                            )
                                            session.add(song_entry)
                                            session.flush()
                                            session.refresh(song_entry)
                                            songid = song_entry.id
                                            session.commit()
                                            eloado = []
                                            szovegirok = []
                                            zeneszerzok = []
                                            megejelnes = 0
                                            for song_info in song_infos:
                                                song_table = BeautifulSoup(
                                                    str(song_info), "lxml"
                                                )
                                                table = song_table.find("table")
                                                data = []
                                                for label in table.select(
                                                    "th, td.attrLabels"
                                                ):
                                                    key = re.sub(
                                                        r"\W+", "", label.text.strip()
                                                    )
                                                    value = (
                                                        label.find_next_sibling().text.strip()
                                                    )

                                                    data.append({key: value})
                                                for d in data:
                                                    if "Szövegírók" in d:
                                                        authors = d["Szövegírók"].split(
                                                            "\n"
                                                        )
                                                        szovegirok.extend(authors)
                                                    if "Előadó" in d:
                                                        performers = d["Előadó"].split(
                                                            "\n"
                                                        )
                                                        eloado.extend(performers)
                                                    if "Zeneszerzők" in d:
                                                        composers = d[
                                                            "Zeneszerzők"
                                                        ].split("\n")
                                                        zeneszerzok.extend(composers)
                                                    if "Megjelenés" in d:
                                                        year = d["Megjelenés"]
                                                        try:
                                                            megjelenes = int(year)
                                                        except Exception as e:
                                                            megjelenes = 0
                                            if not eloado:
                                                eloado = [band_name]
                                            if not szovegirok:
                                                szovegirok = [band_name]
                                            if not zeneszerzok:
                                                zeneszerzok = [band_name]
                                            song2year_entry = Song2Year(
                                                songid=songid, year=megejelnes
                                            )
                                            session.add(song2year_entry)
                                            session.commit()

                                            song2title_entry = Song2Title(
                                                songid=songid, songtitle=song_title
                                            )
                                            session.add(song2title_entry)
                                            session.commit()
                                            # performers
                                            for ea in eloado:
                                                if "(" in ea and ")" in ea:
                                                    startch = ea.index("(")
                                                    endch = ea.index(")")
                                                    aka = ea[startch + 1 : endch]
                                                    ea = ea.replace("(", "")
                                                    ea = ea.replace(")", "")
                                                    ea = ea.replace(aka, "")
                                                else:
                                                    aka = "None"
                                                # performer
                                                performer_query = (
                                                    session.query(Performer)
                                                    .filter_by(name=ea)
                                                    .first()
                                                )
                                                if not performer_query:
                                                    performer_entry = Performer(
                                                        name=ea, aka=aka
                                                    )
                                                    session.add(performer_entry)
                                                    session.flush()
                                                    session.refresh(performer_entry)
                                                    performer_id = performer_entry.id
                                                    session.commit()
                                                else:
                                                    performer_id = performer_query.id
                                                # preformer2song
                                                performer2song_entry = Performer2Song(
                                                    songid=songid,
                                                    performerid=performer_id,
                                                )
                                                session.add(performer2song_entry)
                                                session.commit()
                                                for m2a in members2add:
                                                    person2performer_entry = Person2Performer(
                                                        personid=m2a,
                                                        performerid=performer_id,
                                                    )
                                                    session.add(person2performer_entry)
                                                    session.commit()
                                            # composers
                                            for zs in zeneszerzok:
                                                if "(" in zs and ")" in zs:
                                                    startch = zs.index("(")
                                                    endch = zs.index(")")
                                                    aka = zs[startch + 1 : endch]
                                                    ea = zs.replace("(", "")
                                                    ea = zs.replace(")", "")
                                                    ea = zs.replace(aka, "")
                                                else:
                                                    aka = "None"
                                                # composer
                                                zeneszerzo_query = (
                                                    session.query(Composer)
                                                    .filter_by(name=zs)
                                                    .first()
                                                )
                                                if not zeneszerzo_query:
                                                    zeneszerzo_entry = Composer(
                                                        name=zs, aka=aka
                                                    )
                                                    session.add(zeneszerzo_entry)
                                                    session.flush()
                                                    session.refresh(zeneszerzo_entry)
                                                    zeneszerzo_id = zeneszerzo_entry.id
                                                    session.commit()
                                                else:
                                                    zeneszerzo_id = zeneszerzo_query.id
                                                # composer2song
                                                composer2song_entry = Composer2Song(
                                                    songid=songid,
                                                    composerid=zeneszerzo_id,
                                                )
                                                session.add(composer2song_entry)
                                                session.commit()

                                                # authors
                                            for author in szovegirok:
                                                if "(" in author and ")" in author:
                                                    startch = author.index("(")
                                                    endch = author.index(")")
                                                    aka = author[startch + 1 : endch]
                                                    author = author.replace("(", "")
                                                    author = author.replace(")", "")
                                                    author = author.replace(aka, "")
                                                else:
                                                    aka = "None"
                                                author_query = (
                                                    session.query(Author)
                                                    .filter_by(name=author)
                                                    .first()
                                                )

                                                if not author_query:
                                                    author_entry = Author(
                                                        name=author, aka=aka
                                                    )
                                                    session.add(author_entry)
                                                    session.flush()
                                                    session.refresh(author_entry)
                                                    author_id = author_entry.id
                                                    session.commit()
                                                else:
                                                    author_id = author_query.id

                                                # author2song
                                                author2song_entry = Author2Song(
                                                    songid=songid, authorid=author_id
                                                )
                                                session.add(author2song_entry)
                                                session.commit()

                                        except Exception as e:
                                            print("Song data", e)
                        except Exception as e:
                            print("Band data", e)
    except Exception as e:
        print("First try", e)
        session.close()
    else:
        session.close()
