"""Build the scrollytelling essay (``data/essay/``) from the authored narrative.

The essay reuses the dashboard's data + assets + chart helpers (see
:mod:`src.dashboard.build`). The Hungarian prose lives in an editable ``essay.txt``
(``@scene``/``@step`` blocks); :func:`parse_essay` turns it into scenes the
template's vanilla-JS scroll engine consumes. The page is fully self-contained
(inline HTML/JS + self-hosted woff2 + per-decade cloud PNGs) — copy the folder
into the site repo.

Content format (``essay.txt``)::

    # author comment (ignored)
    @scene <id>
      figure: topics-line        # scene config (figure type + defaults)
      metric: share
    @step
      highlight: Szerelem, Pénz  # step state (drives the pinned figure)
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
    """JSON-serialise the parsed scenes for the ``/*__SCENES__*/`` placeholder.

    Examples:
        >>> scenes_js([])
        '[]'
    """
    return json.dumps(scenes, ensure_ascii=False)


def _kwic_html(kwic: dict[str, Any], word: str, decade: Any) -> str:
    """Render a CLASSIC keyword-aligned concordance for ``word`` in ``decade``.

    Three columns (left context · keyword · right context); the CSS right-aligns
    the left column and centres the keyword so it forms a vertical column.
    """
    items = (kwic.get(word) or {}).get(str(decade)) or []
    if not items:
        return ""
    rows = "".join(
        f'<div class="kw-row">'
        f'<span class="kw-pre">{_html.escape(c["pre"])}</span>'
        f'<span class="kw-kw">{_html.escape(c["kw"])}</span>'
        f'<span class="kw-post">{_html.escape(c["post"])}</span></div>'
        for c in items[:5]
    )
    return (
        f'<div class="kwic"><div class="kwic-h">„{_html.escape(word)}" '
        f"a korszak dalaiban</div>{rows}</div>"
    )


def _kwic_block(kwic: dict[str, Any], words: Any, decade: Any) -> str:
    """Render one classic concordance per requested word (``words`` is a list)."""
    if not words:
        return ""
    if isinstance(words, str):
        words = [words]
    return "".join(_kwic_html(kwic, w, decade) for w in words)


def _img_caption(name: str, credits: dict[str, Any]) -> str:
    """Build a one-line ``<figcaption>`` credit for ``name`` from the manifest."""
    c = credits.get(name) or {}
    if not c:
        return ""
    bits = [b for b in (c.get("author"), c.get("license")) if b]
    line = " · ".join(_html.escape(b) for b in bits)
    src = c.get("source")
    if src:
        line += (
            f' · <a href="{_html.escape(src)}" target="_blank" '
            f'rel="noopener">forrás</a>'
        )
    return f"<figcaption>{line}</figcaption>" if line else ""


def render_body(
    scenes: list[dict[str, Any]],
    kwic: dict[str, Any] | None = None,
    *,
    images: set[str] | None = None,
    credits: dict[str, Any] | None = None,
) -> str:
    """Server-render the scene scaffold (prose + KWIC readable without JS).

    Each scene gets an era-theme class + optional title/subtitle heading + optional
    era image; prose-only scenes (``figure: none``) render as a flat column, chart
    scenes render the sticky-graphic + scrolling-steps structure. A step's
    ``kwic`` state injects that word's concordances (for the scene's ``decade``).

    Args:
        scenes: Parsed scenes (see :func:`parse_essay`).
        kwic: Concordances keyed ``{word: {decade_str: [...]}}``.
        images: Filenames that actually exist in ``assets/images/`` — a scene's
            ``image`` renders only if present (``None`` ⇒ emit all, era skins
            otherwise carry imageless scenes gracefully).
        credits: Image-credit manifest keyed by filename (drives figcaptions).

    Examples:
        >>> html = render_body([
        ...   {"id": "i", "config": {"figure": "none", "theme": "intro"},
        ...    "steps": [{"state": {}, "html": "<p>Hi.</p>"}]},
        ...   {"id": "t", "config": {"figure": "topics-line", "theme": "t1960",
        ...    "title": "X"}, "steps": [{"state": {}, "html": "<p>Go.</p>"}]},
        ... ])
        >>> 'class="scene scene-prose theme-intro"' in html
        True
        >>> 'class="scene scrolly theme-t1960"' in html and 'id="fig-1"' in html
        True
        >>> '<h2>X</h2>' in html
        True
        >>> render_body([{"id": "s", "config": {"image": "x.jpg"},
        ...   "steps": []}], images=set()).count("era-img")  # absent ⇒ skipped
        0
    """
    kwic = kwic or {}
    credits = credits or {}
    out: list[str] = []
    for si, scene in enumerate(scenes):
        cfg = scene.get("config", {})
        figure = cfg.get("figure", "none")
        decade = cfg.get("decade")
        cls = "scene-prose" if figure == "none" else "scrolly"
        if cfg.get("theme"):
            cls += f" theme-{cfg['theme']}"
        head = ""
        if cfg.get("title"):
            head = f'<header class="scene-head"><h2>{_html.escape(cfg["title"])}</h2>'
            if cfg.get("subtitle"):
                head += f'<p class="scene-sub">{_html.escape(cfg["subtitle"])}</p>'
            head += "</header>"
        img = cfg.get("image")
        if img and (images is None or img in images):
            alt = _html.escape((credits.get(img) or {}).get("title", ""))
            head += (
                f'<figure class="era-img"><img src="assets/images/'
                f'{_html.escape(img)}" alt="{alt}" loading="lazy" />'
                f"{_img_caption(img, credits)}</figure>"
            )
        steps = []
        for ti, st in enumerate(scene.get("steps", [])):
            kw = st.get("state", {}).get("kwic")
            extra = _kwic_block(kwic, kw, decade) if kw else ""
            steps.append(
                f'<div class="step" data-scene="{si}" data-step="{ti}">'
                f"{st['html']}{extra}</div>"
            )
        steps_joined = "".join(steps)
        if figure == "none":
            out.append(
                f'<section class="scene {cls}" data-scene="{si}">'
                f"{head}{steps_joined}</section>"
            )
        else:
            out.append(
                f'<section class="scene {cls}" data-scene="{si}">{head}'
                f'<div class="scrolly-grid">'
                f'<div class="scrolly-graphic">'
                f'<div class="figure" id="fig-{si}" role="img"></div>'
                f'<p class="fig-cap" aria-live="polite"></p></div>'
                f'<div class="scrolly-steps">{steps_joined}</div></div></section>'
            )
    return "\n".join(out)


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


def build(
    *,
    db_path: str = "data/music.db",
    corpus_dir: str = "data/processed/corpus",
    topics_dir: Path | str = _dash.DEFAULT_TOPICS_DIR,
    usage_dir: Path | str = _dash.DEFAULT_USAGE_DIR,
    networks_dir: Path | str = _dash.DEFAULT_NETWORKS_DIR,
    emotion_dir: Path | str = _dash.DEFAULT_EMOTION_DIR,
    genre_dir: Path | str = _dash.DEFAULT_GENRE_DIR,
    collab_dir: Path | str = _dash.DEFAULT_COLLAB_DIR,
    rhyme_dir: Path | str = _dash.DEFAULT_RHYME_DIR,
    content_path: Path | str = DEFAULT_CONTENT,
    out_path: Path | str = DEFAULT_OUT,
) -> Path:
    """Build the scrollytelling essay HTML into ``out_path`` (self-contained folder).

    Reuses the dashboard's data + assets + chart helpers; the narrative comes from
    ``content_path`` (``essay.txt``).

    Returns:
        The written HTML path.
    """
    data, keyness, frequency = _dash.load_all_data(
        db_path=db_path,
        corpus_dir=corpus_dir,
        topics_dir=topics_dir,
        usage_dir=usage_dir,
        networks_dir=networks_dir,
        emotion_dir=emotion_dir,
        genre_dir=genre_dir,
        collab_dir=collab_dir,
        rhyme_dir=rhyme_dir,
    )
    data["kwic"] = load_kwic()  # keyword-in-context concordances (essay-only)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Same fonts + cloud PNGs + og card as the dashboard; annotates keyness/
    # frequency rows in place (shared refs in `data`) before serialisation.
    _dash.emit_assets(out_path.parent, keyness, frequency)
    images, credits = _emit_images(out_path.parent)

    scenes = parse_essay(Path(content_path).read_text(encoding="utf-8"))
    charts = _charts_region(_dash._TEMPLATE.read_text(encoding="utf-8"))
    body = render_body(scenes, data.get("kwic"), images=images, credits=credits)

    html = _dash.render_html(data, _TEMPLATE.read_text(encoding="utf-8"))
    html = html.replace("/*__SCENES__*/", scenes_js(scenes))
    html = html.replace("/*__CHARTS__*/", charts)
    html = html.replace("<!--__BODY__-->", body)
    html = html.replace("__SITE_URL__", ESSAY_SITE_URL)
    out_path.write_text(html, encoding="utf-8")
    return out_path


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
    ap.add_argument("--db", default="data/music.db")
    ap.add_argument("--corpus-dir", default="data/processed/corpus")
    ap.add_argument("--content", default=str(DEFAULT_CONTENT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = build(
        db_path=args.db,
        corpus_dir=args.corpus_dir,
        content_path=Path(args.content),
        out_path=Path(args.out),
    )
    print(f"Essay written -> {out}")


if __name__ == "__main__":
    main()
