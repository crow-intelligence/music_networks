"""Tests for the scrollytelling essay builder."""

from __future__ import annotations

import string
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.essay.build import _charts_region, parse_essay, render_body

_prose = st.text(
    alphabet=string.ascii_letters + " áéíóöőúüű", min_size=1, max_size=40
).filter(lambda s: s.strip())


@given(st.lists(_prose, min_size=1, max_size=6))
def test_parse_essay_preserves_steps(proses):
    """Every authored step survives parsing with non-empty rendered prose."""
    body = "@scene s\n  figure: topics-line\n"
    for p in proses:
        body += "@step\n" + p.strip() + "\n"
    scenes = parse_essay(body)
    assert len(scenes) == 1
    assert scenes[0]["config"]["figure"] == "topics-line"
    assert len(scenes[0]["steps"]) == len(proses)
    for st_ in scenes[0]["steps"]:
        assert st_["html"].startswith("<p>") and st_["html"].endswith("</p>")


def test_parse_essay_comments_and_state():
    """``#`` lines are ignored; indented directives become scene/step state."""
    scenes = parse_essay(
        "# a comment\n@scene clouds\n  figure: cloud\n  kind: distinctive\n"
        "@step\n  decade: 1990\nSzöveg.\n"
    )
    assert scenes[0]["config"] == {"figure": "cloud", "kind": "distinctive"}
    assert scenes[0]["steps"][0]["state"] == {"decade": 1990}


def test_render_body_marks_scene_types():
    """Prose scenes render flat; chart scenes render the sticky scrolly structure."""
    scenes = parse_essay(
        "@scene intro\n  figure: none\n@step\nSzia.\n"
        "@scene t\n  figure: topics-line\n@step\nGörbe.\n"
    )
    html = render_body(scenes)
    assert 'class="scene scene-prose" data-scene="0"' in html
    assert 'class="scene scrolly" data-scene="1"' in html
    assert 'id="fig-1"' in html and 'data-step="0"' in html


def test_charts_region_extracts_helpers_from_dashboard():
    """The shared helper region between the dashboard markers is non-empty + valid."""
    from src.dashboard import build as dash

    region = _charts_region(dash._TEMPLATE.read_text(encoding="utf-8"))
    assert "function multiLine" in region
    assert "function barList" in region  # hoisted into the shared region
    assert "__CHARTS_START__" not in region and "__CHARTS_END__" not in region


@pytest.mark.skipif(
    not Path("data/music.db").exists(), reason="needs the local data/ artifacts"
)
def test_build_smoke(tmp_path):
    """End-to-end build produces a self-contained folder with resolved placeholders."""
    from src.essay.build import build

    out = build(out_path=tmp_path / "index.html")
    assert out.exists()
    assert (tmp_path / "assets" / "og.png").exists()
    assert list((tmp_path / "assets" / "clouds").glob("*.png"))
    assert list((tmp_path / "assets" / "fonts").glob("*.woff2"))
    html = out.read_text(encoding="utf-8")
    placeholders = ("/*__DATA__*/", "/*__SCENES__*/", "/*__CHARTS__*/",
                    "<!--__BODY__-->", "__SITE_URL__")
    for placeholder in placeholders:
        assert placeholder not in html
    assert "function multiLine" in html and 'class="scene scrolly' in html
