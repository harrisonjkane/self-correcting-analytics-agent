"""Keep the Streamlit Community Cloud demo awake.

Streamlit sleeps apps after a stretch with no real viewer sessions, and plain HTTP
pings don't count. This opens the app in a headless browser, clicks the wake-up
button if the app is asleep, and holds a live session for a few seconds.

Run by .github/workflows/keep-awake.yml on a schedule.
"""
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

URL = "https://multi-agent-query-engine.streamlit.app"
WAKE_BUTTON = "Yes, get this app back up"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=60_000)

    button = page.get_by_role("button", name=WAKE_BUTTON)
    try:
        button.wait_for(state="visible", timeout=15_000)
    except PlaywrightTimeout:
        print("App is awake.")
    else:
        button.click()
        try:
            button.wait_for(state="hidden", timeout=60_000)
        except PlaywrightTimeout:
            browser.close()
            sys.exit("Clicked the wake button, but the app did not start within 60s.")
        print("App was asleep; wake-up triggered.")

    page.wait_for_timeout(20_000)  # hold a live session so it registers as a visit
    browser.close()
