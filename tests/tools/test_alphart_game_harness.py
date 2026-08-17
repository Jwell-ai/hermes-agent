"""Regression coverage for the 2D game artifact upload harness."""

from types import SimpleNamespace
from unittest.mock import patch

from tools.alphart_tools import (
    CANVAS_GENERATE_GAME_SCHEMA,
    _game_artifact_harness_feedback,
    _game_runtime_harness_feedback,
    _prepare_game_html_for_upload,
    _request_game_upload_target,
)


VALID_GAME = """<!DOCTYPE html>
<html><head><style>
html,body{margin:0;overflow:hidden}#stage{width:1920px;height:1080px;transform:scale(1)}
</style></head><body>
<main id="stage"><canvas id="playfield"></canvas><button id="start">Start</button></main>
<script>
let score = 0;
document.getElementById('start').addEventListener('click', () => {
  score += 1;
  document.getElementById('start').textContent = `Score ${score}`;
});
</script></body></html>"""


def test_game_artifact_harness_accepts_a_complete_self_contained_2d_game():
    assert _game_artifact_harness_feedback(VALID_GAME) == ""


def test_game_fit_wrapper_normalizes_an_unscaled_game_before_validation():
    unscaled_game = VALID_GAME.replace(
        "#stage{width:1920px;height:1080px;transform:scale(1)}", "#stage{}"
    )

    prepared = _prepare_game_html_for_upload(unscaled_game)

    assert "#alphart-game-fit-stage{position:absolute" in prepared
    assert "translate(-50%,-50%) scale(" in prepared
    assert _game_artifact_harness_feedback(prepared) == ""


def test_game_fit_wrapper_replaces_unmarked_legacy_scale_layouts():
    prepared = _prepare_game_html_for_upload(VALID_GAME)

    assert "alphart-game-fit-stage" in prepared
    assert "__ALPHART_GAME_FIT__" in prepared
    assert "#alphart-game-fit-content > #stage" in prepared
    assert "transform:none!important" in prepared


def test_game_tool_requires_a_complete_html_artifact_in_the_agent_flow():
    parameters = CANVAS_GENERATE_GAME_SCHEMA["parameters"]

    assert parameters["required"] == ["prompt", "html"]
    assert "html" in parameters["properties"]
    assert "artifact_dir" not in parameters["properties"]
    assert "artifact_path" not in parameters["properties"]
    assert "files" not in parameters["properties"]


def test_game_artifact_harness_rejects_external_dependencies():
    html = VALID_GAME.replace("</head>", '<script src="https://cdn.example/game.js"></script></head>')

    assert "external scripts" in _game_artifact_harness_feedback(html)


def test_game_artifact_harness_requires_a_playfield_and_real_interaction():
    without_canvas = VALID_GAME.replace('<canvas id="playfield"></canvas>', "")
    without_handler = VALID_GAME.replace(".addEventListener", ".removeEventListener")

    assert "Canvas or SVG" in _game_artifact_harness_feedback(without_canvas)
    assert "real event handler" in _game_artifact_harness_feedback(without_handler)


def test_game_artifact_harness_rejects_layout_that_can_escape_the_stage():
    html = VALID_GAME.replace("#stage{", "#dialog{position:fixed}#stage{")

    assert "position:fixed" in _game_artifact_harness_feedback(html)


def test_game_runtime_harness_uses_browserless_when_configured(monkeypatch):
    monkeypatch.setenv("ALPHART_EDU_BROWSERLESS_URL", "http://browserless:3000")
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return SimpleNamespace(status_code=200, text='{"ok":true}', json=lambda: {"ok": True})

    with patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        assert _game_runtime_harness_feedback(VALID_GAME) == ""

    assert captured["url"] == "http://browserless:3000/function"
    assert captured["json"]["context"]["html"] == VALID_GAME
    assert captured["json"]["context"]["assets"] == {}
    assert "page.setViewport" in captured["json"]["code"]
    assert "page.setRequestInterception(true)" in captured["json"]["code"]
    assert "connect-src 'none'" in captured["json"]["code"]
    assert "__ALPHART_GAME_TEST__" in captured["json"]["code"]
    assert "game stage is clipped by the viewport" in captured["json"]["code"]
    assert "exceed the game stage bounds" in captured["json"]["code"]
    assert '{ name: "tablet", width: 1024, height: 768 }' in captured["json"]["code"]
    assert "visibleGameElements" in captured["json"]["code"]
    assert "alphart-game.local" in captured["json"]["code"]
    assert "Access-Control-Allow-Origin" in captured["json"]["code"]


def test_game_runtime_harness_passes_local_artifacts_to_browserless(monkeypatch):
    monkeypatch.setenv("ALPHART_EDU_BROWSERLESS_URL", "http://browserless:3000")
    captured = {}

    def fake_post(_url, **kwargs):
        captured["json"] = kwargs["json"]
        return SimpleNamespace(status_code=200, text='{"ok":true}', json=lambda: {"ok": True})

    with patch("tools.alphart_tools.requests.post", side_effect=fake_post):
        assert _game_runtime_harness_feedback(
            VALID_GAME,
            {"game.js": {"body": "Y29uc29sZS5sb2coJ3JlYWR5Jyk7", "content_type": "text/javascript"}},
        ) == ""

    assert captured["json"]["context"]["assets"]["game.js"]["content_type"] == "text/javascript"


def test_game_runtime_harness_returns_browserless_violations(monkeypatch):
    monkeypatch.setenv("ALPHART_EDU_BROWSERLESS_URL", "http://browserless:3000")
    response = SimpleNamespace(
        status_code=200,
        text='{"ok":false}',
        json=lambda: {"ok": False, "violations": ["mobile: document scroll/overflow detected"]},
    )

    with patch("tools.alphart_tools.requests.post", return_value=response):
        feedback = _game_runtime_harness_feedback(VALID_GAME)

    assert "mobile: document scroll/overflow detected" in feedback


def test_game_upload_target_preserves_explicit_org_scope():
    captured = {}

    def fake_post(_url, **kwargs):
        captured["json"] = kwargs["json"]
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"url": "https://example.test/games/team-a/session-1/game-1/index.html"},
        )

    with (
        patch("tools.alphart_tools._backend_url", return_value="http://edu"),
        patch("tools.alphart_tools._auth_token", return_value=""),
        patch("tools.alphart_tools._service_token", return_value="agent-token"),
        patch("tools.alphart_tools._ctx", return_value={"org_no": "team-a"}),
        patch("tools.alphart_tools.requests.post", side_effect=fake_post),
    ):
        _request_game_upload_target({"session_id": "session-1"})

    assert captured["json"]["org_no"] == "team-a"
