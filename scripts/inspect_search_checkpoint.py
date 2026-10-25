"""Inspect Research Search page at 1440x900 and 390x844 using Chrome CDP."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import requests
import websockets

OUTPUT_DIR = "artifacts/visual-qa"


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
    path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    res = await send_cdp(ws, "Page.captureScreenshot", {"format": "png"}, msg_id)
    b64_data = res.get("result", {}).get("data")
    if b64_data:
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64_data))
        size = os.path.getsize(path)
        print(f"Captured: {path} ({size:,} bytes)")
        return path
    raise RuntimeError(f"Failed to capture {filename}")


async def get_element_metrics(ws, selectors: dict[str, str], msg_id_start: int = 400) -> dict:
    js_query = f"""
    (function() {{
        var selectors = {json.dumps(selectors)};
        var results = {{}};
        for (var key in selectors) {{
            var el = document.querySelector(selectors[key]);
            if (el) {{
                var rect = el.getBoundingClientRect();
                results[key] = {{
                    found: true,
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height),
                    top: Math.round(rect.top),
                    bottom: Math.round(rect.bottom),
                    visible: rect.width > 0 && rect.height > 0
                }};
            }} else {{
                results[key] = {{ found: false }};
            }}
        }}
        return results;
    }})()
    """
    return await evaluate_js(ws, js_query, msg_id=msg_id_start)


async def inspect_search():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pages = requests.get("http://localhost:9222/json").json()
    page = next(p for p in pages if p.get("type") == "page")
    print(f"Connecting to Chrome page: {page['webSocketDebuggerUrl']}")

    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=25_000_000) as ws:
        await send_cdp(ws, "Page.enable")
        await send_cdp(ws, "Runtime.enable")

        # Navigate to clean search page
        await send_cdp(ws, "Page.navigate", {"url": "http://localhost:8501/?nav=Research+Search"})
        await asyncio.sleep(4)

        selectors = {
            "team_banner": ".team-branding-banner",
            "hero_title": "h1",
            "search_input": '[data-testid="stTextInput"] input',
            "run_button": '.stButton button, [data-testid="stBaseButton-primary"]',
            "mobile_nav": ".st-key-mobile_nav_bar",
            "workflow_cards": ".workflow-cue-card",
        }

        # -------------------------------------------------------------
        # 1. Desktop 1440x900
        # -------------------------------------------------------------
        print("\n--- VIEWPORT 1: Desktop 1440x900 ---")
        await set_viewport(ws, 1440, 900, is_mobile=False, msg_id=101)
        await asyncio.sleep(1.5)

        desktop_scroll_w = await evaluate_js(ws, "document.documentElement.scrollWidth", msg_id=102)
        desktop_inner_w = await evaluate_js(ws, "window.innerWidth", msg_id=103)
        desktop_overflow = desktop_scroll_w > desktop_inner_w
        desktop_metrics = await get_element_metrics(ws, selectors, msg_id_start=104)

        print(f"Desktop: scrollWidth={desktop_scroll_w}, innerWidth={desktop_inner_w}, overflow={desktop_overflow}")
        print("Desktop Element Bounding Boxes:", json.dumps(desktop_metrics, indent=2))

        desktop_screenshot = await capture_screenshot(ws, "01_desktop_research_search_1440x900.png", msg_id=105)

        # -------------------------------------------------------------
        # 2. Simulated Mobile 390x844 (iPhone 12/13/14)
        # -------------------------------------------------------------
        print("\n--- VIEWPORT 2: Simulated Mobile 390x844 ---")
        await set_viewport(ws, 390, 844, is_mobile=True, msg_id=201)
        await asyncio.sleep(1.5)

        mobile_scroll_w = await evaluate_js(ws, "document.documentElement.scrollWidth", msg_id=202)
        mobile_inner_w = await evaluate_js(ws, "window.innerWidth", msg_id=203)
        mobile_overflow = mobile_scroll_w > mobile_inner_w
        mobile_metrics = await get_element_metrics(ws, selectors, msg_id_start=204)

        # Check mobile bottom nav overlap
        nav_overlap_check = await evaluate_js(
            ws,
            """(function() {
                var nav = document.querySelector('.st-key-mobile_nav_bar');
                var footer = document.querySelector('.app-footer');
                var spacer = document.querySelector('.mobile-bottom-spacer');
                return {
                    nav_height: nav ? nav.offsetHeight : 0,
                    spacer_height: spacer ? spacer.offsetHeight : 0,
                    has_spacer: !!spacer
                };
            })()""",
            msg_id=205,
        )

        print(f"Mobile: scrollWidth={mobile_scroll_w}, innerWidth={mobile_inner_w}, overflow={mobile_overflow}")
        print("Mobile Element Bounding Boxes:", json.dumps(mobile_metrics, indent=2))
        print("Mobile Bottom Nav Metrics:", json.dumps(nav_overlap_check, indent=2))

        mobile_screenshot = await capture_screenshot(ws, "03_mobile_research_search_390x844.png", msg_id=206)

        # Summary output
        report = {
            "desktop": {
                "viewport": "1440x900",
                "scrollWidth": desktop_scroll_w,
                "innerWidth": desktop_inner_w,
                "horizontal_overflow": desktop_overflow,
                "screenshot": desktop_screenshot,
                "elements": desktop_metrics,
            },
            "mobile": {
                "viewport": "390x844",
                "scrollWidth": mobile_scroll_w,
                "innerWidth": mobile_inner_w,
                "horizontal_overflow": mobile_overflow,
                "screenshot": mobile_screenshot,
                "elements": mobile_metrics,
                "nav_overlap": nav_overlap_check,
            },
        }

        with open(os.path.join(OUTPUT_DIR, "search_checkpoint_summary.json"), "w") as f:
            json.dump(report, f, indent=2)

        print("\nCheckpoint inspection complete! Summary written to search_checkpoint_summary.json.")


if __name__ == "__main__":
    asyncio.run(inspect_search())
