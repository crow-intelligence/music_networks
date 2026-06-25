"""Scrollytelling essay — a narrative companion to the lyrics dashboard.

A self-contained static page (``data/essay/index.html`` + ``assets/``) that reuses
the dashboard's data pipeline, chart helpers, design tokens, and fonts. The
narrative is authored in :mod:`~src.essay` ``essay.txt`` (``@scene``/``@step``
blocks) and parsed at build time into scroll-driven scenes.
"""
