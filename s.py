import os
import json
import math
import random
import ast
import operator as op
import re
from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# -------------------------
# PDF setup
# -------------------------
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# -------------------------
# OpenAI setup
# -------------------------
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_data(show_spinner=False)
def call_ai(prompt: str) -> str:
    """Fast GPT-4o Mini call with timeout & fallback"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 🚀 Fast & cheap
            messages=[{"role": "user", "content": prompt}],
            timeout=5  # Faster response
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "AI unavailable — please review the concept manually."

def ai_concept(qtext: str) -> str:
    return call_ai(f"Explain briefly the concept tested and how to approach: {qtext}")

def ai_explain(qtext: str, correct: Any) -> str:
    return call_ai(f"Explain step-by-step how to solve: {qtext}. Final answer: {correct}")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Easy Math Tutor Pro", page_icon="🧮", layout="wide")
st.title("🧮 Easy Math Tutor — Pro")
st.caption("Practice • AI hints • Calculator • Notepad • TTS • PDF Markcard")

# -------------------------
# Files & persistence
# -------------------------
ROOT = os.getcwd()
FONTS_DIR = os.path.join(ROOT, "fonts")
DEJAVU_PATH = os.path.join(FONTS_DIR, "DejaVuSans.ttf")
PROGRESS_FILE = os.path.join(ROOT, "progress.json")

if not os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f)

def load_progress():
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_progress(d):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

progress = load_progress()

# -------------------------
# Models & helpers
# -------------------------
@dataclass
class Question:
    prompt: str
    answer: Any
    parse: Callable[[str], Any]
    explain: str
    desc: str = ""

def parse_int(s: str) -> int:
    return int(float(str(s).strip()))

def parse_float2(s: str) -> float:
    return round(float(str(s).strip()), 2)

# Safe calculator using AST
_OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv
}
_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}

def safe_eval(expr: str):
    node = ast.parse(expr, mode="eval").body
    def _eval(n):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Invalid constant")
        if isinstance(n, ast.Num):
            return n.n
        if isinstance(n, ast.BinOp):
            if type(n.op) not in _OPERATORS:
                raise ValueError("Operator not allowed")
            return _OPERATORS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            if type(n.op) not in _UNARY:
                raise ValueError("Unary op not allowed")
            return _UNARY[type(n.op)](_eval(n.operand))
        raise ValueError("Expression not allowed")
    return _eval(node)

# Sanitize text for PDF output
_RE_HIGH_UNICODE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
def sanitize_for_pdf(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = _RE_HIGH_UNICODE.sub("", s)
    s = "".join(ch for ch in s if ch >= " " or ch == "\n")
    return s

# -------------------------
# Question generators
# -------------------------
def q_arith_addsub(diff: str) -> Question:
    ranges = {"Basic": (0, 50), "Intermediate": (50, 500), "Advanced": (500, 5000)}
    lo, hi = ranges[diff]
    a, b = random.randint(lo, hi), random.randint(lo, hi)
    op = random.choice(["+", "-"])
    ans = a + b if op == "+" else a - b
    return Question(prompt=f"Compute: {a} {op} {b}", answer=ans, parse=parse_int, explain=f"{a} {op} {b} = {ans}", desc="Addition/Subtraction")

def q_quadratic(diff: str) -> Question:
    r1, r2 = random.randint(-6,6), random.randint(-6,6)
    a = 1 if diff=="Basic" else random.choice([1,2,3])
    b = -a*(r1+r2); c = a*r1*r2
    choice = random.choice(["sum","product","discriminant"])
    if choice == "sum":
        return Question(prompt=f"For {a}x^2 + {b}x + {c} = 0, sum of roots = ?", answer=r1+r2, parse=parse_int, explain=f"Sum = -b/a = {-b}/{a}", desc="Quadratics: sum")
    if choice == "product":
        return Question(prompt=f"For {a}x^2 + {b}x + {c} = 0, product of roots = ?", answer=r1*r2, parse=parse_int, explain=f"Product = c/a = {c}/{a}", desc="Quadratics: product")
    D = b*b - 4*a*c
    return Question(prompt=f"Discriminant of {a}x^2 + {b}x + {c} = ?", answer=D, parse=parse_int, explain=f"D = b^2 - 4ac = {D}", desc="Quadratics: discriminant")

CURRICULUM = {
    "School": {"Arithmetic": {"Addition/Subtraction": q_arith_addsub}},
    "PU": {"Algebra": {"Quadratics": q_quadratic}},
    "Engineering": {"Algebra": {"Quadratics": q_quadratic}},
}
DIFFICULTIES = ["Basic","Intermediate","Advanced"]

# -------------------------
# Session state defaults
# -------------------------
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "descs" not in st.session_state:
    st.session_state.descs = {}
if "notes" not in st.session_state:
    st.session_state.notes = {}

# -------------------------
# Sidebar — login & choose
# -------------------------
with st.sidebar:
    st.header("👤 Student")
    user = st.text_input("Name or email", value=os.getenv("USER",""))
    user = user.strip() if user else ""
    if not user:
        st.info("Enter your name or email to track progress.")
        st.stop()
    if user not in progress:
        progress[user] = {"records": []}

    st.header("📚 Choose")
    level = st.selectbox("Level", list(CURRICULUM.keys()))
    subject = st.selectbox("Subject", list(CURRICULUM[level].keys()))
    lesson = st.selectbox("Lesson", list(CURRICULUM[level][subject].keys()))
    difficulty = st.selectbox("Difficulty", DIFFICULTIES, index=1)
    num_q = st.slider("Questions", 3, 10, 5)

st.markdown(f"### {level} → {subject} → {lesson}  ·  *{difficulty}*")
st.markdown("Select difficulty then press **Generate Quiz**")

# -------------------------
# Generate quiz
# -------------------------
gen = CURRICULUM[level][subject][lesson]
if st.button("📝 Generate Quiz"):
    qs = [gen(difficulty) for _ in range(num_q)]
    st.session_state.quiz = {"user": user, "level": level, "subject": subject, "lesson": lesson, "difficulty": difficulty, "questions": qs}
    descs = {}
    for i,q in enumerate(qs):
        descs[i] = ai_concept(q.prompt)
    st.session_state.descs = descs
    st.session_state.notes = {}

quiz = st.session_state.quiz

# -------------------------
# TTS helper (browser)
# -------------------------
def tts_button(text: str):
    import json
    js_text = json.dumps(text)
    html = f"""
    <button onclick="(function(){{var u=new SpeechSynthesisUtterance({js_text});u.rate=1;u.lang='en-US';window.speechSynthesis.speak(u);}})()"
     style="padding:6px;border-radius:7px;background:#2f9d27;color:white;border:none;cursor:pointer;">🔊 Read</button>
    """
    components.html(html, height=40)

# -------------------------
# Render quiz + form
# -------------------------
if quiz and quiz.get("user")==user:
    st.subheader("🧪 Quiz")
    with st.form("quiz_form", clear_on_submit=False):
        answers = []
        for i,q in enumerate(quiz["questions"]):
            st.markdown(f"**Q{i+1}.** {q.prompt}")
            desc = st.session_state.descs.get(i, q.desc)
            if desc:
                with st.expander("💡 Concept (AI)"):
                    st.write(desc)
            tts_button(q.prompt)
            ans = st.text_input("Your answer", key=f"ans_{i}")
            answers.append(ans)
            with st.expander("🧮 Calculator"):
                expr = st.text_input("Expression (e.g. 3+4*2)", key=f"calc_{i}")
                if expr:
                    try:
                        val = safe_eval(expr)
                        st.success(f"Result: {val}")
                    except Exception as e:
                        st.error(f"Calc error: {e}")
            with st.expander("📝 Notepad"):
                note = st.text_area("Notes / rough work", key=f"note_{i}", value=st.session_state.notes.get(i,""))
                st.session_state.notes[i] = note
            st.markdown("---")
        submitted = st.form_submit_button("✅ Submit")

    if submitted:
        detailed = []
        correct = 0
        for i,(q,u_raw) in enumerate(zip(quiz["questions"], answers)):
            try:
                parsed = q.parse(u_raw)
            except Exception:
                parsed = None
            ok = parsed == q.answer
            if ok:
                correct += 1
            detailed.append((q.prompt, u_raw or "(blank)", q.answer, q.explain, ok))

        pct = round(100 * correct / max(1, len(quiz["questions"])), 1)
        st.success(f"Score: {correct}/{len(quiz['questions'])} ({pct}%)")

        # Save progress
        progress[user]["records"].append({"level": level, "subject": subject, "lesson": lesson, "difficulty": difficulty, "score": correct, "total": len(quiz["questions"])})
        save_progress(progress)

        # Feedback
        st.markdown("### 📋 Feedback")
        for i,(qp,u,a,ex,ok) in enumerate(detailed):
            tag = "✅ Correct" if ok else "❌ Incorrect"
            st.markdown(f"**Q{i+1}.** {qp} — {tag}")
            st.markdown(f"- Your answer: `{u}`")
            st.markdown(f"- Correct: `{a}`")
            if not ok:
                with st.expander("🧠 AI Explanation"):
                    st.write(ai_explain(qp, a))
            else:
                with st.expander("✳️ Reason"):
                    st.write(ex)
            st.markdown("---")

        # -------------------------
        # PDF markcard builder
        # -------------------------
        def build_pdf_bytes():
            if not FPDF_AVAILABLE:
                st.error("fpdf2 not installed. Run: pip install fpdf2")
                return None
            try:
                pdf = FPDF()
                pdf.set_auto_page_break(True, margin=12)
                pdf.add_page()

                use_dejavu = os.path.exists(DEJAVU_PATH) and os.access(DEJAVU_PATH, os.R_OK)
                try:
                    if use_dejavu:
                        pdf.add_font("DejaVu", "", DEJAVU_PATH, uni=True)
                        pdf.set_font("DejaVu", "", 14)
                    else:
                        pdf.set_font("Helvetica", "", 14)
                except Exception:
                    pdf.set_font("Helvetica", "", 14)
                    use_dejavu = False

                pdf.cell(0, 10, sanitize_for_pdf("Easy Math Tutor - Markcard"), ln=True, align="C")
                pdf.ln(4)
                pdf.set_font("DejaVu" if use_dejavu else "Helvetica", "", 12)
                pdf.cell(0, 8, sanitize_for_pdf(f"Student: {user}"), ln=True)
                pdf.cell(0, 8, sanitize_for_pdf(f"Path: {level}/{subject}/{lesson} ({difficulty})"), ln=True)
                pdf.cell(0, 8, sanitize_for_pdf(f"Score: {correct}/{len(quiz['questions'])} ({pct}%)"), ln=True)
                pdf.ln(6)

                pdf.set_fill_color(200, 220, 255)
                pdf.set_font("DejaVu" if use_dejavu else "Helvetica", "B", 11)
                pdf.cell(10, 10, "No", 1, 0, "C", fill=True)
                pdf.cell(80, 10, "Question", 1, 0, "C", fill=True)
                pdf.cell(30, 10, "Your Ans", 1, 0, "C", fill=True)
                pdf.cell(30, 10, "Correct", 1, 0, "C", fill=True)
                pdf.cell(40, 10, "Status", 1, 1, "C", fill=True)

                pdf.set_font("DejaVu" if use_dejavu else "Helvetica", "", 10)
                for idx, (qp, u, a, ex, ok) in enumerate(detailed, start=1):
                    qps = sanitize_for_pdf(qp)
                    us = sanitize_for_pdf(u)
                    as_ = sanitize_for_pdf(a)
                    stat = "Correct" if ok else "Wrong"

                    x_before = pdf.get_x()
                    y_before = pdf.get_y()
                    pdf.multi_cell(10, 10, str(idx), border=1, align="C")
                    pdf.set_xy(x_before + 10, y_before)
                    pdf.multi_cell(80, 10, qps[:200], border=1)
                    pdf.set_xy(x_before + 90, y_before)
                    pdf.cell(30, 10, us[:20], border=1)
                    pdf.cell(30, 10, as_[:20], border=1)
                    pdf.cell(40, 10, stat, border=1, ln=1)

                out = pdf.output(dest="S")
                if isinstance(out, bytearray):
                    return bytes(out)
                elif isinstance(out, str):
                    return out.encode("latin-1", "ignore")
                return out
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                return None

        pdf_bytes = build_pdf_bytes()
        if pdf_bytes:
            st.download_button("Download Markcard (PDF)", data=pdf_bytes, file_name="markcard.pdf", mime="application/pdf")

# -------------------------
# Dashboard
# -------------------------
st.markdown("---")
st.subheader("📊 Dashboard")
records = progress[user]["records"]

if records:
    df = pd.DataFrame(records)
    st.markdown("### Your Progress Table")
    st.dataframe(df)

    # Bar chart of scores
    st.markdown("### Score Chart")
    df_chart = df.copy()
    df_chart["Percentage"] = df_chart["score"] / df_chart["total"] * 100
    st.bar_chart(df_chart[["Percentage"]])

    # Overall statistics
    total_attempts = len(df)
    avg_score = round(df["score"].sum() / df["total"].sum() * 100, 1) if df["total"].sum() else 0
    best_score = df["score"].max()
    worst_score = df["score"].min()

    st.markdown("### Overall Stats")
    st.markdown(f"- Total quizzes attempted: {total_attempts}")
    st.markdown(f"- Average score: {avg_score}%")
    st.markdown(f"- Best score: {best_score}/{df['total'].max()}")
    st.markdown(f"- Worst score: {worst_score}/{df['total'].max()}")
else:
    st.info("No progress yet. Take some quizzes to see your dashboard here!")
