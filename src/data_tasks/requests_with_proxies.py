import traceback
from itertools import cycle

import requests
from fake_useragent import UserAgent
from lxml.html import fromstring

ua = UserAgent()


def get_proxies():
    url = "https://free-proxy-list.net/"
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = set()
    for i in parser.xpath("//tbody/tr"):
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            # Grabbing IP and corresponding PORT
            proxy = ":".join(
                [i.xpath(".//td[1]/text()")[0], i.xpath(".//td[2]/text()")[0]]
            )
            proxies.add(proxy)
    return proxies


def with_proxies(url):
    proxies = get_proxies()
    proxy_pool = cycle(proxies)
    for i in range(0, len(proxies)):
        proxy = next(proxy_pool)
        try:
            header = {"User-Agent": ua.random}
            proxies = {"http": proxy, "https": proxy}
            response = requests.get(url, proxies=proxies, headers=header)
            if response.status_code == 200:
                return response.text
            else:
                continue
        except Exception as e:
            continue
    return None


def request_with_proxy(url):
    try:
        header = {"User-Agent": ua.random}
        response = requests.get(url, headers=header)
        if response.status_code == 200:
            return response.text
        else:
            return with_proxies(url)
    except Exception as e:
        return None
