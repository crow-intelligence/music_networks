"""Async, resumable scraper for zeneszoveg.hu.

Submodules:
    parse: pure HTML-parsing functions (doctested).
    fetch: async HTTP with polite per-domain rate limiting.
    proxies: optional self-healing free-proxy pool.
    crawl: the orchestrator (discover -> fetch pool -> single writer).
    qa: coverage / success-rate reporting.
"""

BASE_URL = "https://www.zeneszoveg.hu/"
