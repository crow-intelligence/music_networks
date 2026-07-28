"""Tests for the scrollytelling essay builder."""

from __future__ import annotations

import string
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.essay.build import (
    expand_steps,
    parse_essay,
    render_body,
    scenes_js,
)

_prose = st.text(
    alphabet=string.ascii_letters + " áéíóöőúüű", min_size=1, max_size=40
).filter(lambda s: s.strip())


@given(st.lists(_prose, min_size=1, max_size=6))
def test_parse_essay_preserves_steps(proses):
    """Every authored step survives parsing with non-empty rendered prose."""
    body = "@scene s\n  figure: none\n"
    for p in proses:
        body += "@step\n" + p.strip() + "\n"
    scenes = parse_essay(body)
    assert len(scenes) == 1
    assert scenes[0]["config"]["figure"] == "none"
    assert len(scenes[0]["steps"]) == len(proses)
    for st_ in scenes[0]["steps"]:
        assert st_["html"].startswith("<p>") and st_["html"].endswith("</p>")


def test_parse_essay_comments_and_state():
    """``#`` lines are ignored; indented directives become scene/step state."""
    scenes = parse_essay(
        "# a comment\n@scene d\n  figure: kwic-concordance\n  decade: 1990\n"
        "@step\n  kwic: szabadság, pénz\nSzöveg.\n"
    )
    assert scenes[0]["config"] == {"figure": "kwic-concordance", "decade": 1990}
    assert scenes[0]["steps"][0]["state"] == {"kwic": ["szabadság", "pénz"]}


def test_expand_steps_fans_out_kwic_words():
    """Without a chart, the kwic list becomes one step per word (prose on step 0)."""
    scenes = expand_steps(
        parse_essay(
            "@scene d\n  figure: kwic-concordance\n  decade: 1990\n"
            "@step\n  kwic: a, b, c\nPróza.\n"
        )
    )
    words = [s["state"]["word"] for s in scenes[0]["steps"]]
    assert words == ["a", "b", "c"]
    assert scenes[0]["steps"][0]["html"].startswith("<p>")
    assert scenes[0]["steps"][1]["html"] == ""


def test_expand_steps_chart_is_step_zero():
    """With a chart, step 0 is the chart (prose rides it), then one step per word."""
    scenes = expand_steps(
        parse_essay(
            "@scene d\n  figure: kwic-concordance\n  decade: 1980\n"
            "  chart: topics-spot:3\n"
            "@step\n  kwic: élet, pénz\nPróza.\n"
        )
    )
    states = [s["state"] for s in scenes[0]["steps"]]
    assert states == [{"chart": "topics-spot:3"}, {"word": "élet"}, {"word": "pénz"}]
    assert scenes[0]["steps"][0]["html"].startswith("<p>")
    assert scenes[0]["steps"][1]["html"] == ""


def test_render_body_marks_scene_types():
    """Prose scenes render a reveal section; decade scenes render the sticky scrolly."""
    scenes = expand_steps(
        parse_essay(
            "@scene intro\n  figure: none\n  kicker: Bevezető\n@step\nSzia.\n"
            "@scene d\n  figure: kwic-concordance\n  decade: 1970\n  title: X\n"
            "@step\n  kwic: pénz\nGörbe.\n"
        )
    )
    html = render_body(
        scenes, kwic={"pénz": {"1970": [{"pre": "a", "kw": "pénz", "post": "b"}]}}
    )
    assert 'class="reveal" data-scene="0"' in html and "Bevezető" in html
    assert "scene scrolly deco-1970" in html and 'id="fig-1"' in html
    assert 'data-step="0"' in html and "„pénz”" in html
    # server-rendered KWIC line (readable without JS)
    assert 'class="kw-kw"' in html


def test_render_body_emits_chart_host():
    """A decade scene with a ``chart`` gets an ``.era-chart`` host in the graphic."""
    scenes = expand_steps(
        parse_essay(
            "@scene d\n  figure: kwic-concordance\n  decade: 1980\n"
            "  chart: topics-spot:3\n"
            "@step\n  kwic: pénz\nGörbe.\n"
        )
    )
    html = render_body(scenes)
    assert 'class="era-chart" id="chart-0"' in html
    assert 'data-chart="topics-spot:3"' in html


def test_render_body_vectorspace_scene():
    """A ``figure: vectorspace`` scene emits the vektortér host in a reveal section."""
    scenes = parse_essay(
        "@scene vectorspace\n  figure: vectorspace\n  title: Vektortér\n"
        "@step\nSzöveg.\n"
    )
    html = render_body(scenes)
    assert 'class="vspace" id="vspace"' in html
    assert 'class="reveal"' in html and "Vektortér" in html


def test_scenes_js_is_slim():
    """``scenes_js`` drops prose HTML, keeping only figure/decade + step state."""
    scenes = expand_steps(
        parse_essay(
            "@scene d\n  figure: kwic-concordance\n  decade: 1970\n"
            "@step\n  kwic: pénz\nGörbe.\n"
        )
    )
    js = scenes_js(scenes)
    assert '"word": "pénz"' in js and '"decade": 1970' in js
    assert "Görbe" not in js  # prose HTML excluded


@pytest.mark.skipif(
    not Path("data/processed/kwic/kwic.json").exists(),
    reason="needs the local KWIC artifact",
)
def test_build_smoke(tmp_path):
    """End-to-end build produces a self-contained folder with resolved placeholders."""
    from src.essay.build import build

    out = build(out_path=tmp_path / "index.html")
    assert out.exists()
    assert (tmp_path / "assets" / "og.png").exists()
    html = out.read_text(encoding="utf-8")
    placeholders = (
        "/*__DATA__*/",
        "/*__CHARTS__*/",
        "/*__SCENES__*/",
        "<!--__BODY__-->",
        "__SITE_URL__",
    )
    for placeholder in placeholders:
        assert placeholder not in html
    assert 'class="scene scrolly' in html and "kwic-figure" in html
    # charts re-inlined + per-decade chart hosts present
    assert "function multiLine" in html and 'class="era-chart"' in html
