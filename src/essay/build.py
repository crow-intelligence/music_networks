"""Build the scrollytelling essay (``data/essay/``) from the authored narrative.

A KWIC-driven data essay in the **Aporia house style** (dark/gold/serif). Each
decade is a sticky keyword-in-context concordance whose keyword advances one scroll
step at a time; the only inlined data is ``data/processed/kwic``. Fonts load from
the Google-Fonts CDN (matching the other Aporia essays), so the output folder holds
just ``index.html`` + ``assets/`` (era images + an OG card) — copy it into the site
repo. The Hungarian prose lives in an editable ``essay.txt`` (``@scene``/``@step``
blocks); :func:`parse_essay` turns it into scenes the vanilla-JS engine consumes.

Content format (``essay.txt``)::

    # author comment (ignored)
    @scene <id>
      figure: kwic-concordance   # none | dia-kwic | kwic-concordance
      decade: 1970               # scene config (drives the pinned KWIC + accent)
      title: Kétforintos dal
    @step
      kwic: pénz, éjszaka        # expanded to one scroll step per keyword
    Prose paragraph(s) for this step. Blank lines split paragraphs;
    markdown [links](https://…) work (via the dashboard's method_html).

The doctested helpers are pure; heavy imports stay inside :func:`build`.
"""

from __future__ import annotations

import html as _html
import json
import re
from pathlib import Path
from typing import Any

from src.dashboard import build as _dash
from src.lyrics.kwic import DEFAULT_KWIC_DIR as _KWIC_DIR
from src.lyrics.kwic import load_kwic

_TEMPLATE = Path(__file__).with_name("template.html")
DEFAULT_CONTENT = Path(__file__).with_name("essay.txt")
DEFAULT_OUT = Path("data/essay/index.html")
# Public canonical URL for the essay (sibling of the dashboard). Deploy there.
ESSAY_SITE_URL = "https://crowintelligence.org/magyar-dalszovegek-essze/"

# Params whose value is a comma-separated list (others are scalars).
_LIST_KEYS = {"highlight", "cats", "seeds", "kwic"}
# An indented ``key: value`` directive line (scene config or step state).
_PARAM = re.compile(r"^\s+([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")


def _coerce(key: str, value: str) -> Any:
    """Coerce a directive value: list for list-keys, int for ``decade``, else str.

    Examples:
        >>> _coerce("decade", "1990")
        1990
        >>> _coerce("highlight", "Szerelem, Pénz")
        ['Szerelem', 'Pénz']
        >>> _coerce("metric", "share")
        'share'
    """
    v = value.strip()
    if key in _LIST_KEYS:
        return [p.strip() for p in v.split(",") if p.strip()]
    if key == "decade":
        try:
            return int(v)
        except ValueError:
            return v
    return v


def parse_essay(text: str) -> list[dict[str, Any]]:
    r"""Parse the essay content into scenes with steps.

    Args:
        text: The ``essay.txt`` body.

    Returns:
        ``[{"id", "config": {...}, "steps": [{"state": {...}, "html": "<p>…</p>"}]}]``
        — scene config from the lines after ``@scene``, step state from the lines
        after ``@step``, and the step's prose rendered to HTML.

    Examples:
        >>> sc = parse_essay(
        ...     "@scene topics\n  figure: topics-line\n"
        ...     "@step\n  highlight: Szerelem\nProse.\n"
        ... )
        >>> sc[0]["id"], sc[0]["config"]["figure"]
        ('topics', 'topics-line')
        >>> sc[0]["steps"][0]["state"]["highlight"]
        ['Szerelem']
        >>> sc[0]["steps"][0]["html"]
        '<p>Prose.</p>'
    """
    method_html = _dash.method_html

    scenes: list[dict[str, Any]] = []
    scene: dict[str, Any] | None = None
    step: dict[str, Any] | None = None
    prose: list[str] = []

    def end_step() -> None:
        nonlocal step, prose
        if scene is not None and step is not None:
            body = "\n".join(prose).strip()
            step["html"] = method_html(body) if body else ""
            scene["steps"].append(step)
        step, prose = None, []

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if line.startswith("@scene"):
            end_step()
            parts = line.split(None, 1)
            sid = parts[1].strip() if len(parts) > 1 else ""
            scene = {"id": sid, "config": {}, "steps": []}
            scenes.append(scene)
            continue
        if line.startswith("@step"):
            end_step()
            step = {"state": {}}
            continue
        param = _PARAM.match(raw)
        if param and not prose:  # header directive (before this step's prose)
            if step is not None:
                target = step["state"]
            else:
                target = scene["config"] if scene else None
            if target is not None:
                target[param.group(1)] = _coerce(param.group(1), param.group(2))
            continue
        if step is not None:  # prose (blank lines before prose are ignored)
            if line == "" and not prose:
                continue
            prose.append(raw)
    end_step()
    return scenes


def scenes_js(scenes: list[dict[str, Any]]) -> str:
    """Serialise a slim scene list for the ``/*__SCENES__*/`` placeholder.

    The scroll engine only needs each scene's ``config`` (figure + decade + chart)
    and the per-step ``state`` (a KWIC word or the ``chart`` marker), so the prose
    HTML is dropped here.

    Examples:
        >>> scenes_js([])
        '[]'
        >>> js = scenes_js([{"id": "s", "config": {"figure": "kwic-concordance",
        ...   "decade": 1970, "chart": "topics"},
        ...   "steps": [{"state": {"word": "pénz"}, "html": "x"}]}])
        >>> '"word": "pénz"' in js and '"chart": "topics"' in js and '"html"' not in js
        True
    """
    slim = [
        {
            "config": {
                "figure": s.get("config", {}).get("figure"),
                "decade": s.get("config", {}).get("decade"),
                "chart": s.get("config", {}).get("chart"),
            },
            "steps": [{"state": st.get("state", {})} for st in s.get("steps", [])],
        }
        for s in scenes
    ]
    return json.dumps(slim, ensure_ascii=False)


def _kwic_ssr(kwic: dict[str, Any], word: str | None, decade: Any) -> str:
    """No-JS fallback for the pinned concordance (JS re-renders on load)."""
    if not word:
        return ""
    lines = (kwic.get(word) or {}).get(str(decade)) or []
    head = f'<p class="kwic-word">a korszak dalaiban: <b>„{_html.escape(word)}”</b></p>'
    if not lines:
        return head + '<p class="kwic-empty">Ebben az évtizedben nincs találat.</p>'
    rows = "".join(
        f'<div class="kw-row"><span class="kw-pre">{_html.escape(c["pre"])}</span>'
        f'<span class="kw-kw">{_html.escape(c["kw"])}</span>'
        f'<span class="kw-post">{_html.escape(c["post"])}</span></div>'
        for c in lines[:4]
    )
    return head + rows


# Hungarian decade-name back-vowel/front-vowel suffix (ötvenes vs hatvanas …).
_DECADE_SUFFIX = {
    1950: "es",
    1960: "as",
    1970: "es",
    1980: "as",
    1990: "es",
    2000: "es",
    2010: "es",
    2020: "as",
}


def _decade_label(decade: int) -> str:
    """Short Hungarian decade label, e.g. ``1960 -> '1960-as'``.

    Examples:
        >>> _decade_label(1960), _decade_label(1970)
        ('1960-as', '1970-es')
    """
    return f"{decade}-{_DECADE_SUFFIX.get(decade, 'es')}"


def _dia_ssr(kwic: dict[str, Any], word: str | None) -> str:
    """No-JS fallback for the overview diachronic KWIC (one line per decade)."""
    by_dec = (kwic.get(word) or {}) if word else {}
    rows = []
    for d in sorted(int(k) for k in by_dec if str(k).isdigit()):
        lines = by_dec.get(str(d)) or []
        if not lines:
            continue
        c = lines[0]
        rows.append(
            f'<div class="dia-row"><div class="dia-dec">{_decade_label(d)}</div>'
            f'<div class="dia-line">{_html.escape(c["pre"])} '
            f"<b>{_html.escape(c['kw'])}</b> {_html.escape(c['post'])}</div></div>"
        )
    return "".join(rows)


def expand_steps(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand each decade scene into a chart step + one step per KWIC keyword.

    A decade scene is authored with a single ``@step`` carrying ``kwic: a, b, c``
    (and the scene may declare ``chart: …``). The pinned graphic morphs
    chart→KWIC: **step 0** shows the era chart (the prose is read here), then steps
    1..N advance the concordance one keyword at a time. Without a ``chart`` the
    prose rides on the first keyword step instead.

    Examples:
        >>> sc = [{"id": "s", "config": {"figure": "kwic-concordance",
        ...   "chart": "emotion"},
        ...   "steps": [{"state": {"kwic": ["a", "b"]}, "html": "<p>P.</p>"}]}]
        >>> ex = expand_steps(sc)
        >>> [s["state"] for s in ex[0]["steps"]]
        [{'chart': 'emotion'}, {'word': 'a'}, {'word': 'b'}]
        >>> ex[0]["steps"][0]["html"]  # prose rides the chart step
        '<p>P.</p>'
        >>> nochart = expand_steps([{"id": "s", "config": {"figure":
        ...   "kwic-concordance"}, "steps": [{"state": {"kwic": ["a"]},
        ...   "html": "<p>P.</p>"}]}])
        >>> [s["state"] for s in nochart[0]["steps"]]
        [{'word': 'a'}]
    """
    out: list[dict[str, Any]] = []
    for scene in scenes:
        cfg = scene.get("config", {})
        steps = scene.get("steps", [])
        if cfg.get("figure") == "kwic-concordance" and steps:
            words = steps[0].get("state", {}).get("kwic") or []
            if isinstance(words, str):
                words = [words]
            prose = steps[0].get("html", "")
            chart = cfg.get("chart")
            if chart:
                new_steps = [{"state": {"chart": chart}, "html": prose}]
                new_steps += [{"state": {"word": w}, "html": ""} for w in words]
            else:
                new_steps = [
                    {"state": {"word": w}, "html": prose if i == 0 else ""}
                    for i, w in enumerate(words)
                ]
            out.append({**scene, "steps": new_steps or [{"state": {}, "html": prose}]})
        else:
            out.append(scene)
    return out


def render_body(
    scenes: list[dict[str, Any]],
    kwic: dict[str, Any] | None = None,
    *,
    images: set[str] | None = None,
    credits: dict[str, Any] | None = None,
) -> str:
    """Server-render the essay body in the Aporia house style.

    ``figure: none`` → a centred ``.reveal`` prose section; ``figure: dia-kwic`` →
    the overview diachronic-KWIC section; ``figure: kwic-concordance`` → a decade
    **sticky-KWIC scrolly** (a pinned concordance graphic + one scroll step per
    keyword). KWIC lines are also server-rendered so the essay reads without JS.

    Args:
        scenes: Expanded scenes (see :func:`expand_steps`).
        kwic: Concordances ``{word: {decade_str: [{pre, kw, post}]}}``.
        images: Filenames present in ``assets/images/`` (``None`` ⇒ emit all).
        credits: Image-credit manifest (unused here; colophon uses it).

    Examples:
        >>> html = render_body([
        ...   {"id": "i", "config": {"figure": "none", "kicker": "Bevezető"},
        ...    "steps": [{"state": {}, "html": "<p>Hi.</p>"}]},
        ...   {"id": "s", "config": {"figure": "kwic-concordance", "decade": 1970,
        ...    "title": "X"},
        ...    "steps": [{"state": {"word": "pénz"}, "html": "<p>Go.</p>"}]},
        ... ], kwic={"pénz": {"1970": [{"pre": "a", "kw": "pénz", "post": "b"}]}})
        >>> 'class="reveal"' in html and 'Bevezető' in html
        True
        >>> 'scene scrolly deco-1970' in html and 'id="fig-1"' in html
        True
        >>> 'step-word' in html and '„pénz”' in html
        True
    """
    kwic = kwic or {}
    out: list[str] = []
    for si, scene in enumerate(scenes):
        cfg = scene.get("config", {})
        figure = cfg.get("figure", "none")
        steps = scene.get("steps", [])
        prose = steps[0]["html"] if steps else ""
        if figure == "none":
            kicker = cfg.get("kicker", "")
            k = f'<p class="kicker">{_html.escape(kicker)}</p>' if kicker else ""
            out.append(
                f'<main class="container"><section class="reveal" data-scene="{si}">'
                f"{k}{prose}</section></main>"
            )
        elif figure == "dia-kwic":
            word = cfg.get("word", "")
            out.append(
                f'<main class="container wide">'
                f'<section class="reveal" data-scene="{si}">'
                f'<p class="kicker">{_html.escape(cfg.get("kicker", ""))}</p>'
                f'<h2 class="section-title">{_html.escape(cfg.get("title", ""))}</h2>'
                f"{prose}"
                f'<div class="dia-kwic" data-word="{_html.escape(word)}">'
                f"{_dia_ssr(kwic, word)}</div></section></main>"
            )
        elif figure == "vectorspace":
            chart = cfg.get("chart")
            extra = (
                f'<div class="era-chart" id="chart-{si}" '
                f'data-chart="{_html.escape(chart)}"></div>'
                if chart
                else ""
            )
            out.append(
                f'<main class="container wide">'
                f'<section class="reveal" data-scene="{si}">'
                f'<p class="kicker">{_html.escape(cfg.get("kicker", ""))}</p>'
                f'<h2 class="section-title">{_html.escape(cfg.get("title", ""))}</h2>'
                f"{prose}"
                f'<div class="vspace" id="vspace" role="img" '
                f'aria-label="A kulcsszavak elmozdulása a szóvektor-térben, '
                f'évtizedről évtizedre."></div>'
                f"{extra}</section></main>"
            )
        elif figure == "kwic-concordance":
            decade = cfg.get("decade")
            kicker = cfg.get("kicker") or (f"{decade}-es évek" if decade else "")
            img = cfg.get("image")
            bg = ""
            if img and (images is None or img in images):
                bg = (
                    f'<div class="scene-bg" aria-hidden="true" '
                    f"style=\"background-image:url('assets/images/{_html.escape(img)}')\">"
                    f"</div>"
                )
            first_word = next(
                (s["state"]["word"] for s in steps if s.get("state", {}).get("word")),
                None,
            )
            chart = cfg.get("chart")
            chart_host = (
                f'<div class="era-chart" id="chart-{si}" '
                f'data-chart="{_html.escape(chart)}"></div>'
                if chart
                else ""
            )
            step_html = []
            for ti, st in enumerate(steps):
                word = _html.escape(st.get("state", {}).get("word", ""))
                wlbl = f'<p class="step-word">„{word}”</p>' if word else ""
                body = st.get("html", "") + wlbl
                step_html.append(
                    f'<div class="step" data-scene="{si}" data-step="{ti}">{body}</div>'
                )
            cls = "scene scrolly" + (f" deco-{decade}" if decade else "")
            out.append(
                f'<section class="{cls}" data-scene="{si}">{bg}'
                f'<div class="scrolly-grid"><div class="scrolly-graphic">'
                f'<div class="scene-head">'
                f'<p class="kicker">{_html.escape(kicker)}</p>'
                f'<h2 class="section-title">'
                f"{_html.escape(cfg.get('title', ''))}</h2></div>"
                f"{chart_host}"
                f'<div class="kwic-figure" id="fig-{si}" role="img" '
                f'aria-label="Kulcsszavak a korszak dalszövegeiben">'
                f"{_kwic_ssr(kwic, first_word, decade)}</div>"
                f'</div><div class="scrolly-steps">{"".join(step_html)}</div>'
                f"</div></section>"
            )
        out.append('<hr class="section-rule">')
    return "\n".join(out)


def _colophon(credits: dict[str, Any]) -> str:
    """The closing colophon: method note, image sources, and a CTA to the dashboard."""
    method = (
        '<main class="container"><section class="reveal">'
        '<p class="kicker">Colophon</p>'
        '<h2 class="section-title">A módszerről</h2>'
        '<p class="method">Az adatokról és az elemzésről a '
        '<a href="https://crowintelligence.org/magyar-dalszovegek/">Dashboardon</a> '
        "tudhatsz meg többet.</p>"
    )
    src_items = []
    for name in sorted(credits):
        c = credits[name] or {}
        bits = [b for b in (c.get("title"), c.get("author"), c.get("license")) if b]
        line = " — ".join(_html.escape(b) for b in bits)
        src = c.get("source")
        if src:
            line += (
                f' · <a href="{_html.escape(src)}" target="_blank" '
                f'rel="noopener">forrás</a>'
            )
        if line:
            src_items.append(f"<li>{line}</li>")
    sources = (
        f'<p class="kicker">Képek</p><ul class="sources">{"".join(src_items)}</ul>'
        if src_items
        else ""
    )
    cta = (
        '<p class="cta-row"><a class="cta-button" '
        'href="https://crowintelligence.org/magyar-dalszovegek/">'
        "Böngészd az interaktív dashboardot →</a></p>"
    )
    return method + sources + cta + "</section></main>"


def _charts_region(template_text: str) -> str:
    r"""Slice the shared chart-helper JS between the dashboard template's markers.

    Examples:
        >>> _charts_region("a\n// __CHARTS_START__ x\nHELPERS\n// __CHARTS_END__\nb")
        'HELPERS'
    """
    start = template_text.index("__CHARTS_START__")
    start = template_text.index("\n", start) + 1
    end = template_text.index("// __CHARTS_END__")
    return template_text[start:end].strip("\n")


def _load_chart_data(
    kwic_dir: Path | str,
    corpus_dir: str,
    *,
    topics_dir: Path | str = _dash.DEFAULT_TOPICS_DIR,
    usage_dir: Path | str = _dash.DEFAULT_USAGE_DIR,
    networks_dir: Path | str = _dash.DEFAULT_NETWORKS_DIR,
    emotion_dir: Path | str = _dash.DEFAULT_EMOTION_DIR,
) -> dict[str, Any]:
    """Assemble the essay's inlined DATA: KWIC + the four per-decade chart datasets.

    Reuses the dashboard's cheap JSON loaders (emotion/usage/topics/networks) and
    replicates its ``MIN_DECADE`` / stoplist / lexdiv scoping — **without touching
    ``data/music.db``** (only JSON artifacts + the corpus, for MATTR).
    """
    mind = _dash.MIN_DECADE
    data: dict[str, Any] = {"kwic": load_kwic(kwic_dir)}
    data["emotion"] = _dash.load_emotion(Path(emotion_dir))

    info = [
        t
        for t in _dash._load_json(Path(topics_dir) / "topic_info.json", [])
        if t.get("topic_id") != -1
    ]
    over_time = [
        t
        for t in _dash._load_json(Path(topics_dir) / "topics_over_time.json", [])
        if t.get("topic_id") != -1 and t.get("decade", 0) >= mind
    ]
    data["topics"] = {"info": info, "over_time": over_time}

    usage = _dash.load_usage(Path(usage_dir))
    from src.lyrics.corpus import load_corpus

    docs = load_corpus(corpus_dir)
    if docs:
        from src.lyrics.diversity import lexical_diversity_by_decade

        lexdiv = lexical_diversity_by_decade(docs)
        for row in usage.get("vocab_stats", []):
            score = lexdiv.get(row.get("decade"))
            if score is not None:
                row["lexdiv"] = round(score, 3)
    data["usage"] = usage

    stoplist = _dash.load_stoplist()
    networks = _dash.load_networks(Path(networks_dir))
    if networks.get("decades"):
        networks["decades"] = [d for d in networks["decades"] if d >= mind]
        networks["graphs"] = {
            k: _dash._filter_graph(v, stoplist)
            for k, v in networks["graphs"].items()
            if int(k) >= mind
        }
    data["networks"] = networks

    from src.lyrics.vectorspace import load_vectorspace

    data["vectorspace"] = load_vectorspace()
    return data


def build(
    *,
    kwic_dir: Path | str = _KWIC_DIR,
    corpus_dir: str = "data/processed/corpus",
    content_path: Path | str = DEFAULT_CONTENT,
    out_path: Path | str = DEFAULT_OUT,
) -> Path:
    """Build the scrollytelling essay HTML into ``out_path`` (self-contained folder).

    Inlines the KWIC concordances plus four per-decade chart datasets (emotion,
    usage/MATTR, topics, networks) via :func:`_load_chart_data`, and re-inlines the
    dashboard's vanilla-SVG chart helpers (``/*__CHARTS__*/``). Fonts come from the
    Google-Fonts CDN (matching the Aporia essays); the build copies only the era
    images + an OG card. No ``data/music.db`` dependency.

    Returns:
        The written HTML path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_chart_data(kwic_dir, corpus_dir)
    images, credits = _emit_images(out_path.parent)
    _render_og(out_path.parent / "assets" / "og.png")

    scenes = expand_steps(parse_essay(Path(content_path).read_text(encoding="utf-8")))
    body = render_body(scenes, data["kwic"], images=images, credits=credits)
    body += _colophon(credits)
    charts = _charts_region(_dash._TEMPLATE.read_text(encoding="utf-8"))

    html = _TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    html = html.replace("/*__CHARTS__*/", charts)
    html = html.replace("/*__SCENES__*/", scenes_js(scenes))
    html = html.replace("<!--__BODY__-->", body)
    html = html.replace("__SITE_URL__", ESSAY_SITE_URL)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _render_og(path: Path) -> None:
    """Render a 1200×630 dark/gold social card for the essay (best-effort)."""
    from PIL import Image, ImageDraw, ImageFont

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1200, 630), "#1a1a1f")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1200, 8], fill="#e8c97e")

    def _font(size: int, bold: bool = False) -> Any:
        for name in (
            "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf",
            "DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    d.text((80, 120), "APORIA · VIZUÁLIS ESSZÉ", font=_font(26), fill="#9a988f")
    d.text((80, 190), "Nekem írod a dalt —", font=_font(78, bold=True), fill="#e8e6e1")
    d.text((80, 285), "neked elemzem", font=_font(78, bold=True), fill="#e8c97e")
    d.text(
        (80, 430),
        "Hét szó, hetven év magyar dalszövege —",
        font=_font(34),
        fill="#e8e6e1",
    )
    d.text((80, 478), "a pancsoló kislánytól Bagiráig.", font=_font(34), fill="#e8e6e1")
    d.text(
        (80, 560),
        "Crow Intelligence · crowintelligence.org",
        font=_font(24),
        fill="#9a988f",
    )
    img.save(path)


# Era-image source dir (downloaded CC/era assets + a credits.json manifest).
_IMAGES_SRC = Path(__file__).with_name("assets") / "images"


def _emit_images(out_dir: Path) -> tuple[set[str], dict[str, Any]]:
    """Copy era images next to the HTML; return (present filenames, credits).

    Reads ``assets/images/credits.json`` for attribution. Missing images are fine
    — the per-decade era skins carry imageless scenes — so only files that exist
    are copied and rendered.
    """
    import shutil

    dst = out_dir / "assets" / "images"
    dst.mkdir(parents=True, exist_ok=True)
    credits: dict[str, Any] = {}
    cred_path = _IMAGES_SRC / "credits.json"
    if cred_path.exists():
        credits = json.loads(cred_path.read_text(encoding="utf-8"))
    present: set[str] = set()
    if _IMAGES_SRC.is_dir():
        for f in sorted(_IMAGES_SRC.iterdir()):
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                shutil.copy2(f, dst / f.name)
                present.add(f.name)
    return present, credits


def main() -> None:
    """CLI: ``python -m src.essay.build``."""
    import argparse

    ap = argparse.ArgumentParser(description="Build the scrollytelling essay.")
    ap.add_argument("--kwic-dir", default=str(_KWIC_DIR))
    ap.add_argument("--corpus-dir", default="data/processed/corpus")
    ap.add_argument("--content", default=str(DEFAULT_CONTENT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = build(
        kwic_dir=Path(args.kwic_dir),
        corpus_dir=args.corpus_dir,
        content_path=Path(args.content),
        out_path=Path(args.out),
    )
    print(f"Essay written -> {out}")


if __name__ == "__main__":
    main()
