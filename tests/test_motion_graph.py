"""Headless browser test of the motion graph page (docs/motion-graph/index.html)."""

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Page, sync_playwright

PAGE = Path(__file__).resolve().parents[1] / "docs" / "motion-graph" / "index.html"
STEPS = 8
WIDTH, HEIGHT = 1920, 1080


# Counts every way a page could advance itself, before the page's own script runs.
COUNT_TIMERS = """
window.__timers = 0;
for (const name of ["setTimeout", "setInterval", "requestAnimationFrame"]) {
  const original = window[name];
  window[name] = (...args) => { window.__timers++; return original.apply(window, args); };
}
"""

# Every visible leaf element: text, controls and SVG shapes, with its box on screen.
VISIBLE_LEAVES = """
() => [...document.body.querySelectorAll("*")]
  .filter((el) => el.childElementCount === 0 && !["SCRIPT", "STYLE"].includes(el.tagName))
  .filter((el) => el.checkVisibility({ opacityProperty: true, visibilityProperty: true }))
  .map((el) => {
    const r = el.getBoundingClientRect();
    return { tag: el.tagName, text: el.textContent.slice(0, 40), left: r.left, top: r.top,
             right: r.right, bottom: r.bottom, width: r.width, height: r.height };
  })
  .filter((r) => r.width > 0 || r.height > 0)
"""


@pytest.fixture
def requests() -> list[str]:
    return []


@pytest.fixture
def page(requests: list[str]) -> Iterator[Page]:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.on("request", lambda request: requests.append(request.url))
        page.add_init_script(COUNT_TIMERS)
        page.goto(PAGE.as_uri())
        page.evaluate("document.fonts.ready")
        yield page
        browser.close()


def wait_for_animations(page: Page) -> None:
    """Wait until every animation of the current step has finished; fails on an endless one."""
    infinite = page.evaluate(
        "document.getAnimations().filter(a => a.effect.getComputedTiming().iterations === Infinity)"
        ".length"
    )
    assert infinite == 0
    page.evaluate("Promise.all(document.getAnimations().map(a => a.finished))")


def current_step(page: Page) -> str:
    return page.locator("[data-testid=step-indicator]").inner_text()


def test_next_back_arrows_and_space_navigate(page: Page) -> None:
    assert current_step(page) == "01 / 08"
    page.get_by_role("button", name="Next").click()
    assert current_step(page) == "02 / 08"
    page.keyboard.press("ArrowRight")
    assert current_step(page) == "03 / 08"
    page.keyboard.press("Space")
    assert current_step(page) == "04 / 08"
    page.keyboard.press("ArrowLeft")
    assert current_step(page) == "03 / 08"
    page.get_by_role("button", name="Back").click()
    assert current_step(page) == "02 / 08"


def test_each_step_plays_once_and_nothing_advances_without_input(page: Page) -> None:
    for step in range(1, STEPS + 1):
        assert current_step(page) == f"{step:02d} / 08"
        wait_for_animations(page)
        page.wait_for_timeout(1500)
        running = page.evaluate(
            "document.getAnimations().filter(a => a.playState === 'running').length"
        )
        assert running == 0
        assert current_step(page) == f"{step:02d} / 08"
        page.keyboard.press("ArrowRight")
    assert current_step(page) == "08 / 08"
    assert page.evaluate("window.__timers") == 0


def test_bottom_right_quarter_stays_empty_in_every_step(page: Page) -> None:
    for step in range(1, STEPS + 1):
        wait_for_animations(page)
        inside = [
            leaf
            for leaf in page.evaluate(VISIBLE_LEAVES)
            if leaf["right"] > WIDTH / 2 and leaf["bottom"] > HEIGHT / 2
        ]
        assert inside == [], f"step {step}: {inside}"
        page.keyboard.press("ArrowRight")


def test_h_hides_and_shows_the_controls(page: Page) -> None:
    controls = page.get_by_role("navigation", name="Steps")
    assert controls.is_visible()
    page.keyboard.press("h")
    assert controls.is_hidden()
    page.keyboard.press("ArrowRight")
    assert current_step(page) == "02 / 08"
    page.keyboard.press("h")
    assert controls.is_visible()


def test_opens_from_disk_without_network_and_with_the_self_hosted_font(
    page: Page, requests: list[str]
) -> None:
    assert [url for url in requests if not url.startswith("file:")] == []
    assert page.evaluate("document.fonts.check('16px \"JetBrains Mono\"')")
    loaded = page.evaluate(
        "[...document.fonts].filter(f => f.family.includes('JetBrains Mono')).map(f => f.status)"
    )
    assert "loaded" in loaded


def test_parameters_shown_match_the_library_defaults(page: Page) -> None:
    shown = ""
    for _ in range(STEPS):
        shown += (
            page.locator("body").inner_text() + (page.locator("svg").text_content() or "") + "\n"
        )
        page.keyboard.press("ArrowRight")
    for parameter in (
        "keep_recent_turns = 10",
        "min_batch_turns = 8",
        "similarity_threshold = 0.8",
        "cluster_every_n_turns = 5",
        "retrieval_token_budget = 1500",
    ):
        assert parameter in shown


def test_colours_come_from_the_design_tokens() -> None:
    html = PAGE.read_text()
    assert re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", html) == []
