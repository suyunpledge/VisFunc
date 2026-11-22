# ruff: noqa: E501  -- the page templates below are HTML, CSS and JS kept on readable lines
"""在真实 Chrome 渲染的网页上评测 screen-text：能否按文字找到可点击的位置。

    python bench/agent_web_eval.py OUT_DIR > web.json     （需要 Playwright、本机 Chrome、svsg[overlays]）

页面是本地写的四类常见界面（后台设置、密集表格、中文页面、深色仪表盘），另把设置页按 150%
缩放再渲染一次。真值由页面 DOM 给出：每个文字节点的文字和像素框，输入框的值。只有图标的
按钮（SVG + aria-label）单独计数，OCR 本来就找不到它们。
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

from svsg.recognition.screen_text import ScreenTextReader, find_text

STYLE = """
body { font-family: system-ui, "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0; color: #1f2328;
       background: #f6f8fa; font-size: 14px; }
header { display: flex; align-items: center; gap: 18px; padding: 10px 24px; background: #24292f; }
header a { color: #f0f6fc; text-decoration: none; }
main { padding: 20px 32px; }
.card { background: white; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px 20px; margin: 0 0 16px; }
label { display: inline-block; width: 160px; color: #57606a; }
input { width: 320px; padding: 5px 8px; border: 1px solid #d0d7de; border-radius: 6px; font: inherit; }
.row { margin: 10px 0; }
button { font: inherit; padding: 5px 14px; border-radius: 6px; border: 1px solid #d0d7de; background: #f6f8fa; }
button.primary { background: #1f883d; color: white; border-color: #1a7f37; }
.icon { width: 32px; height: 32px; padding: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
td, th { border-bottom: 1px solid #d8dee4; padding: 5px 10px; text-align: left; }
"""
SVG = (
    '<svg width="18" height="18" viewBox="0 0 24 24"><path d="{d}" fill="none" stroke="#57606a" '
    'stroke-width="2"/></svg>'
)
ICONS = {
    "Refresh": "M4 12a8 8 0 0 1 14-5l2-2v6h-6l2-2a6 6 0 1 0 1 7",
    "Delete": "M5 7h14M9 7V4h6v3M7 7l1 13h8l1-13",
    "Share": "M12 3v12M7 8l5-5 5 5M5 14v6h14v-6",
}


def icons() -> str:
    return "".join(
        f'<button class="icon" aria-label="{n}">{SVG.format(d=d)}</button>'
        for n, d in ICONS.items()
    )


def settings(zh: bool = False) -> str:
    t = (lambda en, cn: cn) if zh else (lambda en, cn: en)
    rows = [
        (t("Display name", "显示名称"), t("Morgan Lee", "李明")),
        (t("Email", "电子邮箱"), "morgan.lee@example.com"),
        (t("Company", "公司"), t("Northwind Traders", "北风贸易有限公司")),
        (t("Phone", "电话"), "+1 (415) 555-0132"),
    ]
    form = "".join(f'<div class="row"><label>{a}</label><input value="{b}"></div>' for a, b in rows)
    nav = "".join(
        f"<a href='#'>{x}</a>"
        for x in [
            t("Overview", "概览"),
            t("Repositories", "仓库"),
            t("Settings", "设置"),
            t("Billing", "账单"),
        ]
    )
    return f"""<header>{nav}</header><main>
      <h1>{t("Account settings", "账户设置")}</h1>
      <div class="card">{icons()}<h2>{t("Profile", "个人资料")}</h2>{form}</div>
      <div class="card"><h2>{t("Notifications", "通知")}</h2>
        <div class="row"><input type="checkbox" checked style="width:auto"> {t("Email me about new sign-ins", "有新登录时发邮件提醒")}</div>
        <div class="row"><input type="checkbox" style="width:auto"> {t("Weekly summary", "每周摘要")}</div></div>
      <div><button>{t("Cancel", "取消")}</button> <button class="primary">{t("Save changes", "保存更改")}</button></div>
    </main>"""


def table() -> str:
    names = [
        "Alvarez",
        "Brooks",
        "Chen",
        "Dubois",
        "Evans",
        "Fischer",
        "Garcia",
        "Haddad",
        "Ito",
        "Jensen",
        "Kowalski",
        "Larsen",
        "Moreau",
    ]
    body = "".join(
        f"<tr><td>#{5120 + i}</td><td>{names[i % 13]}</td><td>{['Open', 'Shipped', 'Refunded', 'On hold'][i % 4]}</td>"
        f"<td>${(i * 53) % 900 + 19}.{i % 10}0</td><td>2026-11-{(i % 27) + 1:02d}</td><td>{['DHL', 'UPS', 'FedEx'][i % 3]}</td></tr>"
        for i in range(25)
    )
    return f"""<header><a href='#'>Orders</a><a href='#'>Customers</a><a href='#'>Reports</a></header><main>
      <div class="card">{icons()}<input value="status:open carrier:ups"> <button>Search</button>
      <table><tr><th>Order</th><th>Customer</th><th>Status</th><th>Total</th><th>Updated</th><th>Carrier</th></tr>{body}</table>
      <div class="row">Showing 25 of 1,204 orders <button>Previous</button> <button>Next</button></div></div></main>"""


def dashboard() -> str:
    cards = "".join(
        f'<div class="card" style="display:inline-block;width:260px;margin-right:16px">'
        f'<div>{a}</div><div style="font-size:28px;font-weight:600">{b}</div><div>{c}</div></div>'
        for a, b, c in [
            ("Active users", "18,240", "+4.2% this week"),
            ("Revenue", "$92,410", "-1.3% this week"),
            ("Open tickets", "37", "12 urgent"),
            ("Uptime", "99.97%", "last 30 days"),
        ]
    )
    dark = (
        "body{background:#0d1117;color:#e6edf3} .card{background:#161b22;border-color:#30363d} "
        "button{background:#21262d;color:#e6edf3;border-color:#30363d}"
    )
    return f"""<style>{dark}</style><header><a href='#'>Dashboard</a><a href='#'>Alerts</a></header><main>
      <h1>Service health</h1>{cards}<div class="card">{icons()}<h2>Recent incidents</h2>
      <div class="row">Nov 21 - API latency above 800 ms for 14 minutes</div>
      <div class="row">Nov 18 - Login failures in eu-west-2 resolved</div>
      <button>View all incidents</button></div></main>"""


PAGES = {
    "web-settings": (settings, {}, 1),
    "web-orders": (table, {}, 1),
    "web-settings-zh": (settings, {"zh": True}, 1),
    "web-dashboard-dark": (dashboard, {}, 1),
    "web-settings-150pct": (settings, {}, 1.5),
}

TRUTH_JS = """() => {
  const out = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode, text = node.textContent.trim();
    if (!text || node.parentElement.closest('style')) continue;
    const range = document.createRange(); range.selectNodeContents(node);
    const r = range.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) out.push({text, box: [r.left, r.top, r.right, r.bottom], kind: 'text'});
  }
  for (const el of document.querySelectorAll('input:not([type=checkbox])')) {
    const r = el.getBoundingClientRect(); out.push({text: el.value, box: [r.left, r.top, r.right, r.bottom], kind: 'input'});
  }
  for (const el of document.querySelectorAll('button[aria-label]')) {
    const r = el.getBoundingClientRect(); out.push({text: el.getAttribute('aria-label'), box: [r.left, r.top, r.right, r.bottom], kind: 'icon'});
  }
  return out;
}"""


def main() -> None:
    from playwright.sync_api import sync_playwright

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    reader = ScreenTextReader()
    report = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        for name, (build, kw, scale) in PAGES.items():
            page = browser.new_page(
                viewport={"width": 1920, "height": 1080}, device_scale_factor=scale
            )
            page.set_content(f"<style>{STYLE}</style>{build(**kw)}")
            png = out / f"{name}.png"
            page.screenshot(path=str(png))
            items = [
                dict(i, box=[round(v * scale) for v in i["box"]]) for i in page.evaluate(TRUTH_JS)
            ]
            page.close()
            reader.read(png)  # warm this image size
            start = time.perf_counter()
            result = reader.read(png)
            seconds = time.perf_counter() - start
            texts = [i for i in items if i["kind"] != "icon"]
            counts = Counter(i["text"] for i in texts)
            reach = unique = unique_ok = 0
            misses = []
            for item in texts:
                box = item["box"]
                hits = find_text(result, item["text"], exact=True)
                ins = [
                    h
                    for h in hits
                    if box[0] - 2 <= h["center_px"][0] <= box[2] + 2
                    and box[1] - 2 <= h["center_px"][1] <= box[3] + 2
                ]
                reach += bool(ins)
                if not ins and len(misses) < 6:
                    near = find_text(result, item["text"])
                    misses.append(
                        {"truth": item["text"], "best": near[0]["text"] if near else None}
                    )
                if counts[item["text"]] == 1:
                    unique += 1
                    unique_ok += bool(hits) and hits[0] in ins
            icons_found = sum(
                bool(find_text(result, i["text"], exact=True)) for i in items if i["kind"] == "icon"
            )
            report[name] = {
                "size_px": result["size_px"],
                "read_s": round(seconds, 2),
                "texts": len(texts),
                "reachable": reach,
                "unique": unique,
                "unique_first_hit_correct": unique_ok,
                "icon_buttons": sum(i["kind"] == "icon" for i in items),
                "icon_buttons_found_by_ocr": icons_found,
                "misses": misses,
            }
        browser.close()
    json.dump(report, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
