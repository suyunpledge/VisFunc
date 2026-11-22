# ruff: noqa: E501  -- the page template below is HTML, CSS and JS kept on readable lines
"""Chrome 网页任务：用 Playwright 打开一个独立的 Chrome 窗口（不碰本机浏览器配置），页面状态可核对。

    from agent_web_task import open_page, page_state

页面标题 "Keystone web task"：姓名、邮箱输入框，套餐下拉框，座位数和只有图标的 +/- 按钮
（aria-label），同意条款复选框，"Create account" 提交按钮。提交后页面显示结果，并把状态放在
window.__state 里供核对。窗口用 Win32 设为置顶且不激活，避免被其他窗口挡住。
"""

from __future__ import annotations

TITLE = "Keystone web task"
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Keystone web task</title>
<style>
body { font-family: system-ui, "Segoe UI", sans-serif; margin: 0; background: #f6f8fa; color: #1f2328; font-size: 15px; }
main { max-width: 560px; margin: 32px auto; background: white; border: 1px solid #d0d7de; border-radius: 10px; padding: 24px 28px; }
label { display: block; margin: 14px 0 4px; color: #57606a; }
input[type=text], input[type=email], select { width: 100%; box-sizing: border-box; padding: 7px 9px; font: inherit;
  border: 1px solid #d0d7de; border-radius: 6px; }
.seats { display: flex; align-items: center; gap: 10px; margin-top: 6px; }
.seats button { width: 34px; height: 34px; border-radius: 6px; border: 1px solid #d0d7de; background: #f6f8fa; }
.terms { margin: 16px 0; display: flex; gap: 8px; align-items: center; color: #1f2328; }
button.primary { background: #1f883d; color: white; border: 1px solid #1a7f37; border-radius: 6px; padding: 8px 18px; font: inherit; }
#result { margin-top: 16px; font-weight: 600; }
</style></head><body><main>
<h1>Create a team account</h1>
<label for="name">Full name</label><input id="name" type="text">
<label for="email">Email</label><input id="email" type="email">
<label for="plan">Plan</label>
<select id="plan"><option>Basic</option><option>Pro</option><option>Team</option></select>
<label>Seats</label>
<div class="seats">
  <button id="minus" aria-label="Remove seat"><svg width="14" height="14" viewBox="0 0 14 14"><path d="M2 7h10" stroke="#57606a" stroke-width="2"/></svg></button>
  <span id="count">1</span>
  <button id="plus" aria-label="Add seat"><svg width="14" height="14" viewBox="0 0 14 14"><path d="M2 7h10M7 2v10" stroke="#57606a" stroke-width="2"/></svg></button>
</div>
<div class="terms"><input id="terms" type="checkbox"><label for="terms" style="margin:0;color:inherit">I agree to the terms</label></div>
<button class="primary" id="submit">Create account</button>
<div id="result" role="status"></div>
</main>
<script>
let seats = 1;
window.__state = {submitted: null, attempts: 0};
const count = document.getElementById('count');
document.getElementById('plus').onclick = () => { seats = Math.min(10, seats + 1); count.textContent = seats; };
document.getElementById('minus').onclick = () => { seats = Math.max(1, seats - 1); count.textContent = seats; };
document.getElementById('submit').onclick = () => {
  const s = window.__state; s.attempts += 1;
  const name = document.getElementById('name').value.trim(), email = document.getElementById('email').value.trim();
  const plan = document.getElementById('plan').value, terms = document.getElementById('terms').checked;
  const result = document.getElementById('result');
  if (!name || !email.includes('@')) { result.textContent = 'Please enter a name and a valid email.'; return; }
  if (!terms) { result.textContent = 'Please agree to the terms first.'; return; }
  s.submitted = {name, email, plan, seats, terms};
  result.textContent = `Account created for ${name}: ${plan} plan, ${seats} seats.`;
};
</script></body></html>"""


def open_page(playwright, *, force_accessibility: bool = True):
    """Launch a separate Chrome window with the task page; returns (browser, page)."""
    args = ["--window-position=60,60", "--window-size=1000,860"]
    if force_accessibility:
        args.append("--force-renderer-accessibility")
    browser = playwright.chromium.launch(channel="chrome", headless=False, args=args)
    page = browser.new_page(no_viewport=True)
    page.set_content(PAGE)
    page.wait_for_timeout(500)
    keep_on_top(TITLE)
    return browser, page


def keep_on_top(title: str) -> None:
    import ctypes

    from svsg.recognition.ui_elements import find_window

    hwnd = find_window(title)
    # HWND_TOPMOST, no move, no size, no activation.
    ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)


def page_state(page) -> dict:
    return page.evaluate(
        "() => ({...window.__state, result: document.getElementById('result').textContent,"
        " seats: document.getElementById('count').textContent})"
    )
