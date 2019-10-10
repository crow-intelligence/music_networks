import re
import requests
import bs4
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker

ua = UserAgent()

prefix = "http://www.zeneszoveg.hu/"
base_url = "http://www.zeneszoveg.hu/eloadok/"
initials = "abcdefghijklmopqrstuvwxyz"

stoplist = ["egyuttes/uj.html"]

for ch in initials:
    url = base_url + ch
    header = {"User-Agent": ua.random}
    try:
        html = requests.get(url, headers=header).text
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
                            band_html = requests.get(band_page, headers=header).text
                            band_soup = BeautifulSoup(band_html, "lxml")
                            band_links = band_soup.find_all("a")
                            for band_link in band_links:
                                if band_link["href"].startswith("dalszoveg/"):
                                    if "href" in band_link.attrs:
                                        song_title = band_link.text
                                        header = {"User-Agent": ua.random}
                                        song_url = prefix + band_link["href"]
                                        try:
                                            song_html = requests.get(song_url, headers=header).text
                                            song_soup = BeautifulSoup(song_html, "lxml")
                                            song_text = song_soup.find_all("div", class_="lyrics-plain-text")
                                            song_text = [s.text for s in song_text][0]
                                            song_info = song_soup.find_all("div", {"class": "lyrics-header-text short"})
                                            song_info = [e for e in song_info if "Szövegírók" in e.text]
                                            if len(song_info) == 1:
                                                song_table = BeautifulSoup(str(song_info[0]), "lxml")
                                                table = song_table.find("table")
                                                data = []
                                                for label in table.select(
                                                        'th, td.attrLabels'):
                                                    key = re.sub(r'\W+', '',
                                                                 label.text.strip())
                                                    value = label.find_next_sibling().text.strip()

                                                    data.append({key: value})
                                                print(data)
                                        except Exception as e:
                                            print("Song data", e)
                        except Exception as e:
                            print("Band data", e)
    except Exception as e:
        print("First try", e)

