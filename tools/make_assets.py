"""Regenerates assets/banner.png (GitHub social preview, 1280x640) and assets/favicon.png.
Dev-only: needs `pip install playwright && playwright install chromium`."""
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
LEAF = '<svg viewBox="0 0 24 24"><path fill="{c}" d="M12 2C7 6 4 10 4 14a8 8 0 0 0 16 0c0-4-3-8-8-12z" transform="rotate(25 12 12)"/><path stroke="#150b07" stroke-opacity=".5" stroke-width=".8" d="M12 5v16"/></svg>'


def wave():
    t = np.linspace(0, 1, 700)
    y = 0.55 * np.sin(2 * np.pi * 14 * t) + 0.25 * np.sin(2 * np.pi * 41 * t) * (t > .55) + 0.18 * np.sin(2 * np.pi * 97 * t) * (t > .75)
    pts = " ".join(f"{x * 1180:.1f},{60 - v * 46:.1f}" for x, v in zip(t, y))
    return f'<svg viewBox="0 0 1180 120" width="1180" height="120"><defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#f2a900"/><stop offset=".7" stop-color="#e0662c"/><stop offset="1" stop-color="#d2452b"/></linearGradient></defs><polyline points="{pts}" fill="none" stroke="url(#g)" stroke-width="2.4" stroke-linejoin="round"/></svg>'


leaves = "" and "".join(
    f'<div style="position:absolute;left:{l}px;top:{tp}px;width:{s}px;height:{s}px;opacity:.55;transform:rotate({r}deg)">{LEAF.format(c=c)}</div>'
    for l, tp, s, r, c in [(1040, 60, 70, 20, "#e0662c"), (1130, 170, 44, 140, "#f2a900"), (960, 150, 34, 250, "#b4421f"),
                           (1170, 40, 30, 80, "#d2452b"), (880, 40, 26, 300, "#a4ad3c"), (1090, 300, 52, 200, "#c98a1b")])
BANNER = f"""<html><body style="margin:0;width:1280px;height:640px;background:radial-gradient(900px 500px at 85% -10%,#4a2111,transparent 60%),radial-gradient(700px 500px at -5% 30%,#35170c,transparent 55%),#150b07;
font-family:'Tajawal','Segoe UI',sans-serif;color:#f7e8d0;position:relative;overflow:hidden">
{leaves}
<div style="position:absolute;left:50px;top:70px;display:inline-flex;gap:10px;align-items:center;border:1px solid rgba(242,169,0,.3);border-radius:999px;padding:8px 18px;font-size:20px;letter-spacing:.12em;color:#f2a900;font-weight:700;background:rgba(0,0,0,.3)">
<span style="width:10px;height:10px;border-radius:50%;background:#f2a900"></span>SIMULATED DATA &middot; ESP32-CLASS MODEL</div>
<div style="position:absolute;left:50px;top:150px;font-family:'Fraunces',Georgia,serif;font-weight:800;font-size:150px;line-height:1;letter-spacing:-.02em">Vibe<span style="background:linear-gradient(90deg,#f2a900,#e0662c);-webkit-background-clip:text;color:transparent">Sense</span></div>
<div style="position:absolute;left:54px;top:330px;font-size:34px;color:#c1a68b;max-width:1150px;line-height:1.35">Predictive maintenance from vibration, built for a tiny chip.</div>
<div style="position:absolute;left:50px;top:470px">{wave()}</div>
<div style="position:absolute;left:54px;bottom:24px;font-size:20px;color:#c1a68b">healthy &nbsp;&middot;&nbsp; imbalance &nbsp;&middot;&nbsp; bearing fault &nbsp;&middot;&nbsp; looseness</div>
</body></html>"""

FAV = f"""<html><body style="margin:0;width:128px;height:128px;background:#150b07;display:flex;align-items:center;justify-content:center">
<svg viewBox="0 0 128 128" width="128" height="128"><rect width="128" height="128" rx="28" fill="#24140d"/>
<g transform="translate(24 14) scale(3.3)">{LEAF.format(c="#e0662c")}</g>
<polyline points="14,98 34,98 42,80 52,116 62,86 70,98 114,98" fill="none" stroke="#f2a900" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/></svg></body></html>"""

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 640})
    pg.set_content(BANNER); pg.wait_for_timeout(400)
    pg.screenshot(path=str(ROOT / "assets" / "banner.png"))
    pg = b.new_page(viewport={"width": 128, "height": 128})
    pg.set_content(FAV); pg.screenshot(path=str(ROOT / "assets" / "favicon.png"))
    b.close()
print("assets written")
