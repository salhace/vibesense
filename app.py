"""VibeSense dashboard (Streamlit). Everything here is driven by SIMULATED vibration."""
import time
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from vibesense.features import extract, spectrum
from vibesense.simulator import AR_NAMES, STATES, WINDOW, make_window

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="VibeSense", page_icon=str(ROOT / "assets" / "favicon.png"), layout="wide")

# ---------------------------------------------------------------- text
T = {
    "en": {
        "dir": "ltr", "tag": "Predictive maintenance on a tiny chip",
        "sub": "A small AI model listens to a machine's vibration and warns before it fails. Running on simulated data.",
        "sim": "SIMULATED DATA", "lang": "Language", "inspect": "Inspect machine", "fault": "Simulate condition",
        "sev": "Fault severity", "live": "Live mode", "new": "Capture new window",
        "health": "Health", "wave": "Vibration signal", "wave_s": "1 second of acceleration, in g",
        "spec": "Frequency spectrum", "spec_s": "Which frequencies carry the energy", "probs": "What the model thinks",
        "fleet": "Machine fleet", "age": "Ageing time-lapse", "age_s": "60 simulated days as the fault grows",
        "age_pick": "Fault that develops", "thr": "Maintenance threshold", "action": "Recommended action",
        "ok": "Healthy", "warn": "Watch", "bad": "Critical", "model": "Model accuracy on simulated test windows",
        "model_n": "Real machines are noisier. Treat this as a prototype, not a certified tool.",
        "emb": "Embedded model for ESP32-class chips", "emb_acc": "accuracy",
        "day": "Day", "alert_day": "Health drops below the threshold around day", "no_alert": "Stays above the threshold",
        "states": {"healthy": "Healthy", "imbalance": "Imbalance", "bearing_fault": "Bearing fault", "looseness": "Looseness"},
        "acts": {
            "healthy": "No action needed. Keep monitoring.",
            "imbalance": "Check the rotor or fan blades for dirt or missing weight, then schedule balancing.",
            "bearing_fault": "Inspect the bearing at the next stop. Check lubrication and replace if worn.",
            "looseness": "Check mounting bolts and the machine base for play.",
        },
        "machines": ["Water pump", "Cooling fan", "Air compressor", "Conveyor motor"],
        "x1": "1x shaft", "freq": "Hz",
    },
    "ar": {
        "dir": "rtl", "tag": "صيانة تنبؤية على شريحة صغيرة",
        "sub": "نموذج ذكاء اصطناعي صغير يسمع اهتزاز الآلة وينبه قبل ما تتعطل. يعمل على بيانات محاكاة.",
        "sim": "بيانات محاكاة", "lang": "اللغة", "inspect": "افحص الآلة", "fault": "حاكِ حالة",
        "sev": "شدة العطل", "live": "وضع مباشر", "new": "التقط عينة جديدة",
        "health": "الصحة", "wave": "إشارة الاهتزاز", "wave_s": "ثانية واحدة من التسارع بوحدة g",
        "spec": "طيف الترددات", "spec_s": "أي الترددات تحمل الطاقة", "probs": "ماذا يرى النموذج",
        "fleet": "أسطول الآلات", "age": "تسريع الزمن للتقادم", "age_s": "ستون يوما محاكاة مع تفاقم العطل",
        "age_pick": "العطل الذي يتطور", "thr": "حد الصيانة", "action": "الإجراء المقترح",
        "ok": "سليم", "warn": "راقب", "bad": "حرج", "model": "دقة النموذج على عينات اختبار محاكاة",
        "model_n": "الآلات الحقيقية أكثر ضجيجا. هذا نموذج أولي وليس أداة معتمدة.",
        "emb": "النموذج المدمج لشرائح ESP32", "emb_acc": "الدقة",
        "day": "يوم", "alert_day": "تنزل الصحة تحت الحد تقريبا في اليوم", "no_alert": "تبقى فوق الحد",
        "states": AR_NAMES,
        "acts": {
            "healthy": "لا حاجة لإجراء. استمر بالمراقبة.",
            "imbalance": "افحص الدوّار أو ريش المروحة من الأوساخ أو نقص الوزن ثم جدول موازنة.",
            "bearing_fault": "افحص المحمل في أقرب توقف. تأكد من التشحيم وبدّله إذا تآكل.",
            "looseness": "افحص براغي التثبيت وقاعدة الآلة.",
        },
        "machines": ["مضخة المياه", "مروحة التبريد", "ضاغط الهواء", "محرك الناقل"],
        "x1": "التردد الأساسي", "freq": "هرتز",
    },
}

# ---------------------------------------------------------------- style
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=Tajawal:wght@400;500;700&display=swap');
:root{--bg:#150b07;--bg2:#24140d;--card:rgba(255,236,206,.055);--line:rgba(242,169,0,.20);
--cream:#f7e8d0;--muted:#c1a68b;--amber:#f2a900;--orange:#e0662c;--rust:#b4421f;--maple:#d2452b;--olive:#a4ad3c;}
html,body,.stApp{background:radial-gradient(1200px 700px at 85% -10%,#4a2111 0%,transparent 60%),
radial-gradient(900px 600px at -10% 20%,#35170c 0%,transparent 55%),var(--bg)!important;color:var(--cream);
font-family:'Tajawal','Segoe UI',system-ui,sans-serif;}
header[data-testid="stHeader"],footer,#MainMenu{display:none!important}
.block-container{padding-top:1.4rem;max-width:1280px}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#2a160d,#170c07)!important;border-right:1px solid var(--line)}
section[data-testid="stSidebar"] *{color:var(--cream)!important}
[data-testid="stSelectbox"] [role="group"]{background:#2c170e!important;border:1px solid var(--line)!important;border-radius:12px!important}
[data-testid="stSelectbox"] input{color:var(--cream)!important;-webkit-text-fill-color:var(--cream)!important}
[data-testid="stSelectbox"] svg{color:var(--amber)!important}
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] *{color:var(--cream)!important}
[role="listbox"]{background:#2c170e!important;border:1px solid var(--line)!important}
[role="option"]{color:var(--cream)!important;background:#2c170e!important}
[role="option"][data-focused="true"],[role="option"]:hover{background:#4a2111!important}
div[data-baseweb="select"]>div{background:#2c170e!important;border:1px solid var(--line)!important;border-radius:12px!important}
div[data-baseweb="select"] *{color:var(--cream)!important}
ul[role="listbox"],div[data-baseweb="popover"] li{background:#2c170e!important;color:var(--cream)!important}
.stButton>button{background:linear-gradient(90deg,#e0662c,#b4421f)!important;border:0!important;border-radius:12px!important;color:#fff!important;font-weight:700}
.stButton>button *{color:#fff!important}
h1,h2,h3{font-family:'Fraunces','Tajawal',serif!important;color:var(--cream)!important}
.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:26px;padding:34px 36px;
background:linear-gradient(120deg,rgba(224,102,44,.28),rgba(242,169,0,.10) 55%,rgba(180,66,31,.20));margin-bottom:18px}
.hero h1{font-size:3.1rem;line-height:1;margin:0 0 8px;letter-spacing:-.02em}
.hero h1 b{background:linear-gradient(90deg,var(--amber),var(--orange));-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{color:var(--muted);font-size:1.05rem;margin:0;max-width:640px}
.pill{display:inline-flex;gap:8px;align-items:center;border:1px solid var(--line);border-radius:999px;padding:5px 12px;
font-size:.74rem;letter-spacing:.08em;color:var(--amber);background:rgba(0,0,0,.25);margin-bottom:14px;font-weight:700}
.dot{width:8px;height:8px;border-radius:50%;background:var(--amber);box-shadow:0 0 0 0 rgba(242,169,0,.7);animation:pulse 1.8s infinite}
@keyframes pulse{70%{box-shadow:0 0 0 9px rgba(242,169,0,0)}100%{box-shadow:0 0 0 0 rgba(242,169,0,0)}}
.card{border:1px solid var(--line);background:var(--card);border-radius:22px;padding:20px 22px;backdrop-filter:blur(6px);height:100%}
.card h3{font-size:1.05rem;margin:0 0 2px}.card .s{color:var(--muted);font-size:.82rem;margin-bottom:10px}
.prob{display:flex;align-items:center;gap:10px;margin:9px 0;font-size:.9rem}
.prob .n{width:104px;color:var(--cream)}.prob .bar{flex:1;height:9px;border-radius:9px;background:rgba(255,255,255,.07);overflow:hidden}
.prob .bar i{display:block;height:100%;border-radius:9px;background:linear-gradient(90deg,var(--amber),var(--orange))}
.prob .v{width:44px;text-align:end;color:var(--muted);font-variant-numeric:tabular-nums}
.act{margin-top:14px;padding:12px 14px;border-radius:14px;border:1px dashed var(--line);color:var(--cream);font-size:.92rem;background:rgba(0,0,0,.2)}
.act small{display:block;color:var(--amber);letter-spacing:.08em;font-weight:700;margin-bottom:3px}
.fleet{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:6px 0 18px}
.m{border:1px solid var(--line);border-radius:18px;padding:14px 16px;background:var(--card);position:relative;overflow:hidden}
.m.sel{border-color:var(--amber);box-shadow:0 0 0 1px var(--amber) inset,0 10px 30px rgba(224,102,44,.18)}
.m .t{font-weight:700;font-size:.95rem}.m .st{font-size:.78rem;color:var(--muted);margin-top:2px}
.m .h{font-family:'Fraunces',serif;font-size:2rem;font-weight:800;margin-top:6px}
.m .top{display:flex;justify-content:space-between;align-items:center;gap:8px}.m .tag{font-size:.68rem;font-weight:700;padding:3px 9px;border-radius:99px;white-space:nowrap}
.card svg,.hero svg{direction:ltr}
.foot{color:var(--muted);font-size:.8rem;margin-top:18px;line-height:1.7}
@media(max-width:900px){.fleet{grid-template-columns:repeat(2,1fr)}.hero h1{font-size:2.2rem}}
div[data-testid="stSlider"] [role=slider]{background:var(--amber)!important}
</style>
"""

# ---------------------------------------------------------------- model
@st.cache_resource
def load_model():
    path = ROOT / "models" / "model.joblib"
    if not path.exists() or not (ROOT / 'models' / 'severity.joblib').exists():
        from vibesense import train
        train.main()
    return joblib.load(path)


@st.cache_data
def load_metrics():
    import json
    p = ROOT / "models" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {"accuracy": None}


@st.cache_resource
def load_reg():
    return joblib.load(ROOT / "models" / "severity.joblib")


def predict(clf, x):
    f = extract(x)
    p = clf.predict_proba([f])[0]
    out = {s: float(v) for s, v in zip([STATES[i] for i in clf.classes_], p)}
    out["_sev"] = float(np.clip(load_reg().predict([f])[0], 0, 1))
    return out


def health_of(probs):
    # estimated fault severity (0..1) turned into a 0..100 health score
    return 100.0 * (1.0 - probs["_sev"])


def band(h):
    return ("ok", "#a4ad3c") if h >= 80 else (("warn", "#f2a900") if h >= 55 else ("bad", "#d2452b"))


# ---------------------------------------------------------------- svg
def ring(h, color, label):
    r, c = 70, 2 * np.pi * 70
    off = c * (1 - h / 100)
    return (f'<svg viewBox="0 0 180 180" width="100%" style="max-width:230px;display:block;margin:auto">'
            f'<defs><linearGradient id="rg" x1="0" x2="1"><stop offset="0" stop-color="{color}"/><stop offset="1" stop-color="#f2a900"/></linearGradient></defs>'
            f'<circle cx="90" cy="90" r="{r}" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="14"/>'
            f'<circle cx="90" cy="90" r="{r}" fill="none" stroke="url(#rg)" stroke-width="14" stroke-linecap="round" '
            f'stroke-dasharray="{c:.1f}" stroke-dashoffset="{off:.1f}" transform="rotate(-90 90 90)"/>'
            f'<text x="90" y="92" text-anchor="middle" font-family="Fraunces,serif" font-weight="800" font-size="44" fill="#f7e8d0">{h:.0f}</text>'
            f'<text x="90" y="118" text-anchor="middle" font-size="13" fill="{color}" font-weight="700" letter-spacing="1.5">{label}</text></svg>')


def wave_svg(x):
    W, H, lim = 760, 220, 3.0
    xs = np.asarray(x)[::2]
    n = len(xs)
    pts = [f"{i * W / (n - 1):.1f},{H / 2 - np.clip(v, -lim, lim) / lim * (H / 2 - 8):.1f}" for i, v in enumerate(xs)]
    line = "M" + " L".join(pts)
    area = line + f" L{W},{H / 2} L0,{H / 2} Z"
    grid = "".join(f'<line x1="0" x2="{W}" y1="{y}" y2="{y}" stroke="rgba(255,236,206,.07)"/>' for y in (30, H / 2, H - 30))
    return (f'<svg viewBox="0 0 {W} {H}" width="100%"><defs><linearGradient id="wg" x1="0" x2="1">'
            f'<stop offset="0" stop-color="#f2a900"/><stop offset=".6" stop-color="#e0662c"/><stop offset="1" stop-color="#b4421f"/></linearGradient>'
            f'<linearGradient id="wa" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#e0662c" stop-opacity=".35"/>'
            f'<stop offset="1" stop-color="#e0662c" stop-opacity="0"/></linearGradient></defs>{grid}'
            f'<path d="{area}" fill="url(#wa)"/><path d="{line}" fill="none" stroke="url(#wg)" stroke-width="1.8" stroke-linejoin="round"/></svg>')


def spec_svg(x, t):
    f, m = spectrum(x)
    sel = f <= 500
    f, m = f[sel], m[sel]
    nb = 125
    chunk = len(f) // nb
    vals = [float(m[i * chunk:(i + 1) * chunk].max()) for i in range(nb)]
    top = max(max(vals), 0.4)
    W, H = 760, 220
    bw = W / nb
    bars = []
    for i, v in enumerate(vals):
        h = (v / top) ** 0.5 * (H - 46)
        col = "#f2a900" if v / top < .12 else ("#e0662c" if v / top < .5 else "#d2452b")
        bars.append(f'<rect x="{i * bw + 1:.1f}" y="{H - 22 - h:.1f}" width="{bw - 2:.1f}" height="{max(h, 1.5):.1f}" rx="2" fill="{col}" opacity=".92"/>')
    marks = ""
    for hx, lab in [(50, t["x1"]), (100, "2x"), (150, "3x"), (200, "4x")]:
        xx = hx / 500 * W
        marks += (f'<line x1="{xx}" x2="{xx}" y1="14" y2="{H - 22}" stroke="rgba(247,232,208,.25)" stroke-dasharray="3 5"/>'
                  f'<text x="{xx + 4}" y="24" font-size="11" fill="#c1a68b">{lab}</text>')
    axis = "".join(f'<text x="{v / 500 * W}" y="{H - 5}" font-size="11" fill="#c1a68b" text-anchor="{"start" if v == 0 else ("end" if v == 500 else "middle")}">{v}</text>' for v in (0, 100, 200, 300, 400, 500))
    return f'<svg viewBox="0 0 {W} {H}" width="100%">{marks}{"".join(bars)}{axis}</svg>'


def age_svg(days, health, thr, t):
    W, H, pad = 1000, 250, 34
    n = len(days)
    X = lambda i: pad + i * (W - 2 * pad) / (n - 1)
    Y = lambda v: H - pad - (v / 100) * (H - 2 * pad)
    line = "M" + " L".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(health))
    area = line + f" L{X(n - 1):.1f},{H - pad} L{X(0):.1f},{H - pad} Z"
    ty = Y(thr)
    cross = next((i for i, v in enumerate(health) if v < thr), None)
    cm = ""
    if cross is not None:
        cm = (f'<circle cx="{X(cross):.1f}" cy="{Y(health[cross]):.1f}" r="7" fill="#d2452b" stroke="#f7e8d0" stroke-width="2"/>'
              f'<text x="{X(cross) + 12:.1f}" y="{Y(health[cross]) - 10:.1f}" font-size="13" fill="#f7e8d0" font-weight="700">{t["day"]} {days[cross]}</text>')
    grid = "".join(f'<line x1="{pad}" x2="{W - pad}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="rgba(255,236,206,.07)"/>'
                   f'<text x="6" y="{Y(v) + 4:.1f}" font-size="11" fill="#c1a68b">{v}</text>' for v in (0, 25, 50, 75, 100))
    xl = "".join(f'<text x="{X(i):.1f}" y="{H - 10}" font-size="11" fill="#c1a68b" text-anchor="middle">{days[i]}</text>' for i in range(0, n, 10))
    return (f'<svg viewBox="0 0 {W} {H}" width="100%"><defs><linearGradient id="ag" x1="0" x2="1"><stop offset="0" stop-color="#a4ad3c"/>'
            f'<stop offset=".5" stop-color="#f2a900"/><stop offset="1" stop-color="#d2452b"/></linearGradient>'
            f'<linearGradient id="aa" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#f2a900" stop-opacity=".30"/><stop offset="1" stop-color="#f2a900" stop-opacity="0"/></linearGradient></defs>'
            f'{grid}<line x1="{pad}" x2="{W - pad}" y1="{ty:.1f}" y2="{ty:.1f}" stroke="#d2452b" stroke-dasharray="6 6"/>'
            f'<text x="{W - pad}" y="{ty - 6:.1f}" font-size="12" fill="#d2452b" text-anchor="end">{t["thr"]}</text>'
            f'<path d="{area}" fill="url(#aa)"/><path d="{line}" fill="none" stroke="url(#ag)" stroke-width="3" stroke-linejoin="round"/>{cm}{xl}</svg>')


@st.cache_data
def ageing(state, _model_id=0):
    clf = load_model()
    rng = np.random.default_rng(7)
    days = list(range(0, 61))
    hs = []
    for d in days:
        sev = min(1.0, (d / 60) ** 1.4)
        vals = [health_of(predict(clf, make_window(state, max(sev, 0.02), rng))) for _ in range(5)]
        hs.append(float(np.mean(vals)))
    return days, hs


def html(s):
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


# ---------------------------------------------------------------- app
clf = load_model()
metrics = load_metrics()

def cur_sel():
    """Selected machine index. Streamlit can hand back the displayed label after a language switch, so map it back."""
    v = st.session_state.get("sel", 1)
    if isinstance(v, str):
        idx = next((L["machines"].index(v) for L in T.values() if v in L["machines"]), None)
        return idx if idx is not None else (int(v) if v.isdigit() else 0)
    return int(v)


def cur_cond():
    c = st.session_state.get("cond", "healthy")
    if c in STATES:
        return c
    return next((k for L in T.values() for k, lab in L["states"].items() if lab == c), "healthy")


if "machines" not in st.session_state:
    st.session_state.machines = [("healthy", 0.0), ("imbalance", 0.55), ("healthy", 0.0), ("bearing_fault", 0.45)]
    st.session_state.sel = 1
    st.session_state.tick = 0

lang = st.sidebar.radio("Language / اللغة", ["English", "العربية"], horizontal=True)
t = T["en" if lang == "English" else "ar"]
st.sidebar.markdown("---")


def _load_selected():
    state, sev = st.session_state.machines[cur_sel()]
    st.session_state.cond = state
    st.session_state.sev = float(sev) if sev else 0.5


def _save_selected():
    c = cur_cond()
    st.session_state.machines[cur_sel()] = (c, 0.0 if c == "healthy" else float(st.session_state.sev))


if "cond" not in st.session_state:
    _load_selected()

names = t["machines"]
st.sidebar.selectbox(t["inspect"], range(4), key="sel", format_func=lambda i: names[i], on_change=_load_selected)
st.sidebar.selectbox(t["fault"], STATES, key="cond", format_func=lambda s: t["states"][s], on_change=_save_selected)
st.sidebar.slider(t["sev"], 0.05, 1.0, step=0.05, key="sev", on_change=_save_selected,
                  disabled=(cur_cond() == "healthy"))
live = st.sidebar.toggle(t["live"], value=True)
if st.sidebar.button(t["new"]):
    st.session_state.tick += 1

acc = metrics.get("accuracy")
html(f'<style>.stApp{{direction:{t["dir"]}}}</style>{CSS}')
html(f'''<div class="hero"><div class="pill"><span class="dot"></span>{t["sim"]}</div>
<h1>Vibe<b>Sense</b></h1><p>{t["tag"]}. {t["sub"]}</p></div>''')


def body():
    seed = int(time.time() * 2) if live else st.session_state.tick + 1000
    rng = np.random.default_rng(seed)
    cards, results = [], []
    for i, (s, sev) in enumerate(st.session_state.machines):
        x = make_window(s, sev, rng)
        p = predict(clf, x)
        results.append((x, p))
        h = health_of(p)
        b, col = band(h)
        cards.append(f'<div class="m {"sel" if i == cur_sel() else ""}"><div class="top"><div class="t">{names[i]}</div><span class="tag" style="background:{col}33;color:{col}">{t[b].upper()}</span></div>'
                     f'<div class="st">{t["states"][max((k for k in p if k != "_sev"), key=p.get)]}</div>'
                     f'<div class="h" style="color:{col}">{h:.0f}<span style="font-size:.9rem;color:#c1a68b">/100</span></div></div>')
    html(f'<h3 style="margin:4px 0 8px">{t["fleet"]}</h3><div class="fleet">{"".join(cards)}</div>')

    x, p = results[cur_sel()]
    h = health_of(p)
    b, col = band(h)
    top = max((k for k in p if k != '_sev'), key=p.get)
    c1, c2 = st.columns([1, 2.1], gap="medium")
    probs = "".join(f'<div class="prob"><span class="n">{t["states"][s]}</span><span class="bar"><i style="width:{v * 100:.0f}%"></i></span><span class="v">{v * 100:.0f}%</span></div>' for s, v in sorted(((k, v) for k, v in p.items() if k != '_sev'), key=lambda kv: -kv[1]))
    with c1:
        html(f'<div class="card"><h3>{names[cur_sel()]}</h3><div class="s">{t["health"]}</div>{ring(h, col, t[b].upper())}'
             f'<div style="margin-top:14px"><h3 style="font-size:.95rem">{t["probs"]}</h3>{probs}</div>'
             f'<div class="act"><small>{t["action"].upper()}</small>{t["acts"][top]}</div></div>')
    with c2:
        html(f'<div class="card" style="margin-bottom:14px"><h3>{t["wave"]}</h3><div class="s">{t["wave_s"]}</div>{wave_svg(x)}</div>'
             f'<div class="card"><h3>{t["spec"]}</h3><div class="s">{t["spec_s"]} ({t["freq"]})</div>{spec_svg(x, t)}</div>')


(st.fragment(run_every=1.0)(body) if live else st.fragment(body))()

# ageing time-lapse
st.markdown("")
cur = st.session_state.machines[cur_sel()][0]
default = cur if cur != "healthy" else "bearing_fault"
fl = [s for s in STATES if s != "healthy"]
pick = st.selectbox(t["age_pick"], fl, index=fl.index(default), format_func=lambda s: t["states"][s])
days, hs = ageing(pick)
cross = next((d for d, v in zip(days, hs) if v < 60), None)
msg = f'{t["alert_day"]} <bdi>{cross}</bdi>' if cross is not None else t["no_alert"]
html(f'<div class="card"><h3>{t["age"]}</h3><div class="s">{t["age_s"]} — {msg}</div>{age_svg(days, hs, 60, t)}</div>')

acc_txt = f"{acc * 100:.1f}%" if acc is not None else "n/a"
emb_path = ROOT / "models" / "embedded_metrics.json"
emb = ""
if emb_path.exists():
    import json as _j
    e = _j.loads(emb_path.read_text())
    emb = (f'<br>{t["emb"]}: <bdi><b style="color:#f2a900">{e["trees"]} trees, ~{e["approx_flash_bytes"] / 1024:.0f} KB</b></bdi>, '
           f'{t["emb_acc"]} <bdi><b style="color:#f2a900">{e["accuracy_small_model"] * 100:.0f}%</b></bdi>')
html(f'<div class="foot">{t["model"]}: <bdi><b style="color:#f2a900">{acc_txt}</b></bdi>. {t["model_n"]}{emb}</div>')
