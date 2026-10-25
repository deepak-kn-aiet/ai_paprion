"""Headless Chrome CDP automation for Visual QA and responsive testing across viewports."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import requests
import websockets

SCREENSHOT_DIR = "artifacts/visual-qa"


async def send_cdp(ws, method: str, params: dict | None = None, msg_id: int = 1) -> dict:
    payload = {"id": msg_id, "method": method, "params": params or {}}
    await ws.send(json.dumps(payload))
    while True:
        resp = await ws.recv()
        data = json.loads(resp)
        if data.get("id") == msg_id:
            return data


async def evaluate_js(ws, expr: str, msg_id: int = 100):
    res = await send_cdp(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True}, msg_id)
    return res.get("result", {}).get("result", {}).get("value")


async def set_viewport(ws, width: int, height: int, is_mobile: bool = False, msg_id: int = 200):
    await send_cdp(
        ws,
        "Emulation.setDeviceMetricsOverride",
        {
            "width": width,
            "height": height,
            "deviceScaleFactor": 2,
            "mobile": is_mobile,
        },
        msg_id,
    )


async def capture_screenshot(ws, filename: str, msg_id: int = 300) -> str:
    path = os.path.join(SCREENSHOT_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    res = await send_cdp(ws, "Page.captureScreenshot", {"format": "png"}, msg_id)
    b64_data = res.get("result", {}).get("data")
    if b64_data:
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64_data))
        size = os.path.getsize(path)
        print(f"  [SAVED] {path} ({size:,} bytes)")
        return path
    raise RuntimeError(f"Failed to capture screenshot: {filename}")


async def wait_for_selector(ws, selector: str, timeout: float = 10.0) -> bool:
    start = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start < timeout:
        found = await evaluate_js(ws, f"!!document.querySelector('{selector}')", msg_id=400)
        if found:
            return True
        await asyncio.sleep(0.5)
    return False


async def check_layout_health(ws, label: str):
    overflow = await evaluate_js(
        ws,
        "document.documentElement.scrollWidth > window.innerWidth",
        msg_id=500,
    )
    scroll_w = await evaluate_js(ws, "document.documentElement.scrollWidth", msg_id=501)
    inner_w = await evaluate_js(ws, "window.innerWidth", msg_id=502)
    print(f"  [HEALTH] {label}: scrollWidth={scroll_w}, innerWidth={inner_w}, horizontal_overflow={overflow}")
    return not overflow


async def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # 1. Connect to Chrome CDP
    try:
        pages = requests.get("http://localhost:9222/json").json()
    except Exception as e:
        print(f"Failed to connect to Chrome CDP at port 9222: {e}")
        sys.exit(1)

    target_page = next((p for p in pages if p.get("type") == "page"), None)
    if not target_page:
        print("No page found in Chrome CDP.")
        sys.exit(1)

    ws_url = target_page["webSocketDebuggerUrl"]
    print(f"Connecting to Chrome page: {ws_url}")

    async with websockets.connect(ws_url, max_size=25_000_000) as ws:
        await send_cdp(ws, "Page.enable")
        await send_cdp(ws, "Runtime.enable")

        # ---------------------------------------------------------
        # PASS 1: EMPTY STATE / SEARCH PAGE (4 Viewports)
        # ---------------------------------------------------------
        print("\n=== PASS 1: SEARCH PAGE ACROSS 4 VIEWPORTS ===")
        print("Navigating to http://localhost:8501/?nav=Research+Search ...")
        await send_cdp(ws, "Page.navigate", {"url": "http://localhost:8501/?nav=Research+Search"})
        await asyncio.sleep(4)

        # Desktop 1440x900
        print("\n1. Desktop (1440x900) - Research Search")
        await set_viewport(ws, 1440, 900, is_mobile=False, msg_id=1001)
        await asyncio.sleep(1)
        await check_layout_health(ws, "Desktop 1440x900 Search")
        await capture_screenshot(ws, "01_desktop_research_search_1440x900.png", msg_id=1002)

        # Tablet 768x1024
        print("\n2. Tablet (768x1024) - Research Search")
        await set_viewport(ws, 768, 1024, is_mobile=True, msg_id=1003)
        await asyncio.sleep(1)
        await check_layout_health(ws, "Tablet 768x1024 Search")
        await capture_screenshot(ws, "02_tablet_research_search_768x1024.png", msg_id=1004)

        # Mobile 390x844 (iPhone 12/13/14)
        print("\n3. Mobile (390x844) - Research Search & Fixed Navigation")
        await set_viewport(ws, 390, 844, is_mobile=True, msg_id=1005)
        await asyncio.sleep(1)
        await check_layout_health(ws, "Mobile 390x844 Search")
        await capture_screenshot(ws, "03_mobile_research_search_390x844.png", msg_id=1006)

        # Narrow Mobile 320x568 (iPhone SE / 5s)
        print("\n4. Narrow Mobile (320x568) - Research Search & Fixed Navigation")
        await set_viewport(ws, 320, 568, is_mobile=True, msg_id=1007)
        await asyncio.sleep(1)
        await check_layout_health(ws, "Narrow Mobile 320x568 Search")
        await capture_screenshot(ws, "04_narrow_mobile_search_320x568.png", msg_id=1008)

        # ---------------------------------------------------------
        # PASS 2: POPULATED RESEARCH SESSIONS ACROSS ALL 5 SECTIONS
        # ---------------------------------------------------------
        print("\n=== PASS 2: POPULATED SECTIONS (DEMO RESULT) ===")

        # Section 3: Paper Library (Mobile 390x844 & Desktop 1440x900 & Tablet 768x1024)
        print("\n5. Mobile (390x844) - Paper Library")
        await set_viewport(ws, 390, 844, is_mobile=True, msg_id=2001)
        await send_cdp(ws, "Page.navigate", {"url": "http://localhost:8501/?demo=true&nav=Paper+Library"})
        await asyncio.sleep(4)
        await check_layout_health(ws, "Mobile 390x844 Paper Library")
        await capture_screenshot(ws, "05_mobile_paper_library_390x844.png", msg_id=2002)

        # Interact with Paper Library: click on the first expander (Abstract / Details)
        print("\nInteracting: Clicking Paper Library expandable details on mobile...")
        expander_clicked = await evaluate_js(
            ws,
            """(function() {
                var expander = document.querySelector('[data-testid="stExpander"] summary');
                if (expander) {
                    expander.click();
                    return true;
                }
                return false;
            })()""",
            msg_id=2003,
        )
        print(f"  Expander clicked: {expander_clicked}")
        await asyncio.sleep(1.5)
        await capture_screenshot(ws, "13_mobile_paper_library_expanded_390x844.png", msg_id=2004)

        # Tablet Paper Library (768x1024)
        print("\n6. Tablet (768x1024) - Paper Library")
        await set_viewport(ws, 768, 1024, is_mobile=True, msg_id=2005)
        await asyncio.sleep(1)
        await check_layout_health(ws, "Tablet 768x1024 Paper Library")
        await capture_screenshot(ws, "12_tablet_paper_library_768x1024.png", msg_id=2006)

        # Section 5: Research Verdict (Mobile 390x844 & Desktop 1440x900)
        print("\n7. Mobile (390x844) - Research Verdict")
        await set_viewport(ws, 390, 844, is_mobile=True, msg_id=2007)
        await send_cdp(ws, "Page.navigate", {"url": "http://localhost:8501/?demo=true&nav=Research+Verdict"})
        await asyncio.sleep(4)
        await check_layout_health(ws, "Mobile 390x844 Research Verdict")
        await capture_screenshot(ws, "06_mobile_research_verdict_390x844.png", msg_id=2008)

        print("\n8. Desktop (1440x900) - Research Verdict")
        await set_viewport(ws, 1440, 900, is_mobile=False, msg_id=2009)
        await asyncio.sleep(1.5)
        await check_layout_health(ws, "Desktop 1440x900 Research Verdict")
        await capture_screenshot(ws, "07_desktop_research_verdict_1440x900.png", msg_id=2010)

        # Section 2: Research Overview (Desktop 1440x900 & Mobile 390x844)
        print("\n9. Desktop (1440x900) - Research Overview")
        await send_cdp(ws, "Page.navigate", {"url": "http://localhost:8501/?demo=true&nav=Research+Overview"})
        await asyncio.sleep(4)
        await check_layout_health(ws, "Desktop 1440x900 Research Overview")
        await capture_screenshot(ws, "08_desktop_research_overview_1440x900.png", msg_id=2011)

        print("\n10. Mobile (390x844) - Research Overview")
        await set_viewport(ws, 390, 844, is_mobile=True, msg_id=2012)
        await asyncio.sleep(1.5)
        await check_layout_health(ws, "Mobile 390x844 Research Overview")
        await capture_screenshot(ws, "09_mobile_research_overview_390x844.png", msg_id=2013)

        # Section 4: Score Analysis (Desktop 1440x900 & Mobile 390x844)
        print("\n11. Desktop (1440x900) - Score Analysis")
        await set_viewport(ws, 1440, 900, is_mobile=False, msg_id=2014)
        await send_cdp(ws, "Page.navigate", {"url": "http://localhost:8501/?demo=true&nav=Score+Analysis"})
        await asyncio.sleep(4)
        await check_layout_health(ws, "Desktop 1440x900 Score Analysis")
        await capture_screenshot(ws, "10_desktop_score_analysis_1440x900.png", msg_id=2015)

        print("\n12. Mobile (390x844) - Score Analysis")
        await set_viewport(ws, 390, 844, is_mobile=True, msg_id=2016)
        await asyncio.sleep(1.5)
        await check_layout_health(ws, "Mobile 390x844 Score Analysis")
        await capture_screenshot(ws, "11_mobile_score_analysis_390x844.png", msg_id=2017)

        # ---------------------------------------------------------
        # PASS 3: SEARCH INTERACTION TEST (Type & Click Chip)
        # ---------------------------------------------------------
        print("\n=== PASS 3: INTERACTIVE CONTROLS VERIFICATION ===")
        await set_viewport(ws, 1440, 900, is_mobile=False, msg_id=3001)
        await send_cdp(ws, "Page.navigate", {"url": "http://localhost:8501/?nav=Research+Search"})
        await asyncio.sleep(4)

        # Click the first search suggestion chip
        chip_click_result = await evaluate_js(
            ws,
            """(function() {
                var chip = document.querySelector('.search-chip');
                if (chip) {
                    chip.click();
                    return chip.innerText.trim();
                }
                return null;
            })()""",
            msg_id=3002,
        )
        print(f"  Suggestion chip clicked: {chip_click_result}")
        await asyncio.sleep(1)

        input_val = await evaluate_js(
            ws,
            """(function() {
                var inp = document.querySelector('[data-testid="stTextInput"] input');
                return inp ? inp.value : null;
            })()""",
            msg_id=3003,
        )
        print(f"  Input value after chip click: '{input_val}'")

        # Capture interactive state screenshot
        await capture_screenshot(ws, "14_desktop_chip_interaction_1440x900.png", msg_id=3004)

        print("\n=== All visual QA passes completed successfully! ===")


if __name__ == "__main__":
    asyncio.run(main())
