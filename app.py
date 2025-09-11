import os
import json
import random
import ast
import operator as op
import re
from dataclasses import dataclass
from typing import Any, Callable
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import sympy as sp

# -------------------------
# PDF setup
# -------------------------
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Easy Math Tutor Pro", page_icon="🧮", layout="wide")
st.title("🧮 Easy Math Tutor — Pro")
st.caption("Practice • Calculator • Detailed Solutions • TTS • PDF Markcard")

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
# Helpers
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

# Safe calculator
_OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv
}
_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}

def safe_eval(expr: str):
    expr = expr.replace(",", ".").strip()
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

# -------------------------
# Enhanced step-by-step calculator
# -------------------------
def calc_with_steps(expr: str):
    """
    Evaluate any mathematical expression and provide detailed step-by-step explanation.
    Supports:
    - Implicit multiplication (e.g., 3x -> 3*x)
    - Polynomials
    - Fractions
    - Trigonometric functions
    - Equations (solve for x)
    """
    try:
        expr = expr.replace(",", ".").strip()

        # Insert * for implicit multiplication (e.g., 3x -> 3*x, 2(x+1) -> 2*(x+1))
        expr = re.sub(r"(\d)([a-zA-Z\(])", r"\1*\2", expr)
        expr = re.sub(r"([a-zA-Z\)])(\d)", r"\1*\2", expr)

        steps = [f"Input expression: {expr}"]

        # Check if it's an equation (contains '=')
        if '=' in expr:
            left, right = expr.split('=')
            x = sp.symbols('x')
            eq = sp.Eq(sp.sympify(left), sp.sympify(right))
            steps.append(f"Equation form: {eq}")
            solution = sp.solve(eq, x)
            steps.append(f"Solution: {solution}")
            return solution, "\n".join(steps)
        else:
            sym_expr = sp.sympify(expr)
            steps.append(f"Parsed expression: {sym_expr}")

            expanded = sp.expand(sym_expr)
            if expanded != sym_expr:
                steps.append(f"Expanded: {expanded}")

            simplified = sp.simplify(expanded)
            if simplified != expanded:
                steps.append(f"Simplified: {simplified}")

            numeric_result = simplified.evalf()
            steps.append(f"Numeric evaluation: {numeric_result}")

            return numeric_result, "\n".join(steps)

    except Exception as e:
        return None, f"Error: {e}"
# Sanitize text for PDF
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
    op_symbol = random.choice(["+", "-"])
    ans = a + b if op_symbol=="+" else a - b
    explain = f"Step-by-step:\n{a} {op_symbol} {b} = {ans}"
    return Question(prompt=f"Compute: {a} {op_symbol} {b}", answer=ans, parse=parse_int, explain=explain, desc="Addition/Subtraction")

# -------------------------
# Fraction parser (must be before q_fraction)
# -------------------------
def parse_fraction_input(s: str) -> float:
    """
    Parse user input for fraction questions.
    Accepts decimal or fraction string and returns float rounded to 2 decimals.
    """
    try:
        import sympy as sp
        val = sp.sympify(s)   # Handles '3/4' or '0.75'
        return round(float(val), 2)
    except Exception:
        raise ValueError("Invalid input. Enter a number like 0.75 or a fraction like 3/4.")

# -------------------------
# Fraction question generator
# -------------------------
def q_fraction(diff: str) -> Question:
    a,b,c,d = random.randint(1,10), random.randint(1,10), random.randint(1,10), random.randint(1,10)
    op_symbol = random.choice(["+", "-"])
    val1 = sp.Rational(a,b)
    val2 = sp.Rational(c,d)
    ans = val1+val2 if op_symbol=="+" else val1-val2
    explain = f"{a}/{b} {op_symbol} {c}/{d} = {ans} ≈ {float(ans):.2f}"
    return Question(
        prompt=f"Compute: {a}/{b} {op_symbol} {c}/{d}",
        answer=round(float(ans),2),   # store numeric value for comparison
        parse=parse_fraction_input,   # use parser defined above
        explain=explain,
        desc="Fractions"
    )

def q_trig(diff: str) -> Question:
    theta = sp.symbols('theta')
    funcs = [sp.sin, sp.cos, sp.tan]
    f = random.choice(funcs)
    angle = random.choice([0, 30, 45, 60, 90])
    rad = sp.rad(angle)
    val = float(f(rad).evalf())
    explain = (
        f"Step 1: Identify function and angle → {f.__name__}({angle}°)\n"
        f"Step 2: Convert degrees to radians → {angle}° × π/180 = {rad}\n"
        f"Step 3: Apply the function → {f.__name__}({rad})\n"
        f"Step 4: Evaluate exact value if possible\n"
        f"Step 5: Numeric result → {round(val,4)}"
    )
    return Question(prompt=f"Compute {f.__name__}({angle}°)", answer=round(val,2), parse=parse_float2, explain=explain, desc="Trigonometry")


def q_linear(diff: str) -> Question:
    x = sp.symbols('x')
    a,b,c = random.randint(1,10), random.randint(0,10), random.randint(10,30)
    expr = a*x + b - c
    ans = sp.solve(expr, x)[0]
    explain = f"Linear equation: {a}x + {b} = {c}\nSolve: x = ({c}-{b})/{a} = {int(ans)}"
    return Question(prompt=f"Solve for x: {a}x + {b} = {c}", answer=int(ans), parse=parse_int, explain=explain, desc="Linear Equation")

def q_quadratic(diff: str) -> Question:
    r1,r2 = random.randint(-6,6), random.randint(-6,6)
    a = 1 if diff=="Basic" else random.choice([1,2,3])
    b,c = -a*(r1+r2), a*r1*r2
    x = sp.symbols('x')
    expr = a*x**2 + b*x + c
    roots = sp.solve(expr, x)
    explain = f"Quadratic equation: {a}x^2 + {b}x + {c} = 0\nRoots: {roots}"
    return Question(prompt=f"Solve: {a}x^2 + {b}x + {c} = 0", answer=[float(r) for r in roots], parse=lambda x:[float(e.strip()) for e in x.split(",")], explain=explain, desc="Quadratic Equation")

def q_quadratic_discriminant(diff: str) -> Question:
    a = random.randint(1, 5)
    b = random.randint(-10, 10)
    c = random.randint(-10, 10)
    D = b**2 - 4*a*c
    explain = f"Quadratic: {a}x² + {b}x + {c} = 0\nDiscriminant formula: D = b² - 4ac\nD = ({b})² - 4*{a}*{c} = {D}\n"
    if D > 0:
        explain += "Two distinct real roots"
    elif D == 0:
        explain += "One real root"
    else:
        explain += "No real roots"
    return Question(prompt=f"Compute the discriminant of: {a}x² + {b}x + {c} = 0", answer=D, parse=parse_int, explain=explain, desc="Quadratic Discriminant")

def q_calculus_diff(diff: str) -> Question:
    x = sp.symbols('x')
    funcs = [
        lambda: random.randint(1,5)*x**random.randint(1,4),
        lambda: random.randint(1,5)*sp.sin(random.randint(1,5)*x),
        lambda: random.randint(1,5)*sp.cos(random.randint(1,5)*x),
        lambda: random.randint(1,5)*sp.exp(random.randint(1,3)*x),
        lambda: random.randint(1,5)*sp.log(x + random.randint(1,3))
    ]
    f = random.choice(funcs)()
    deriv = sp.diff(f, x)

    # Step-by-step explanation
    explain = f"Original function: f(x) = {f}\n"
    if f.is_Pow or f.is_Mul:
        explain += f"Apply power/multiplication rules to differentiate.\n"
    elif f.has(sp.sin):
        explain += f"Use derivative formula: d/dx[sin(kx)] = k*cos(kx)\n"
    elif f.has(sp.cos):
        explain += f"Use derivative formula: d/dx[cos(kx)] = -k*sin(kx)\n"
    elif f.has(sp.exp):
        explain += f"Derivative of exp(kx) is k*exp(kx)\n"
    elif f.has(sp.log):
        explain += f"Derivative of log(x+a) is 1/(x+a)\n"
    explain += f"Final derivative: f'(x) = {deriv}"

    return Question(prompt=f"Differentiate: {f} w.r.t x",
                    answer=str(deriv),
                    parse=str,
                    explain=explain,
                    desc="Differentiation")

# -------------------------
# Curriculum
# -------------------------
CURRICULUM = {
    "School": {
        "Arithmetic": {"Addition/Subtraction": q_arith_addsub, "Fractions": q_fraction},
        "Trigonometry": {"Basic Trig": q_trig},
    },
    "PU": {
        "Algebra": {"Linear Equations": q_linear, "Quadratics": q_quadratic, "Discriminant": q_quadratic_discriminant},
        "Trigonometry": {"Trigonometry": q_trig},
        "Calculus": {"Differentiation": q_calculus_diff},
        "Fractions": {"Fractions": q_fraction},
    },
    "Engineering": {
        "Algebra": {"Linear Equations": q_linear, "Quadratics": q_quadratic, "Discriminant": q_quadratic_discriminant},
        "Trigonometry": {"Trigonometry": q_trig},
        "Calculus": {"Differentiation": q_calculus_diff},
        "Fractions": {"Fractions": q_fraction},
    }
}

DIFFICULTIES = ["Basic","Intermediate","Advanced"]

# -------------------------
# Session defaults
# -------------------------
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "notes" not in st.session_state:
    st.session_state.notes = {}

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("👤 Student")
    user = st.text_input("Name or email", value=os.getenv("USER","")).strip()
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
    st.session_state.notes = {}

quiz = st.session_state.quiz

# -------------------------
# TTS helper
# -------------------------
def tts_button(text: str):
    text_js = text.replace('"', '\\"')
    html = f"""
    <button onclick="
        const utter = new SpeechSynthesisUtterance('{text_js}');
        utter.rate = 1;
        utter.lang = 'en-US';
        speechSynthesis.speak(utter);
    " style="padding:6px;border-radius:7px;background:#2f9d27;color:white;border:none;cursor:pointer;">
    🔊 Read
    </button>
    """
    components.html(html, height=40)

# -------------------------
# Quiz form
# -------------------------
if quiz and quiz.get("user")==user:
    st.subheader("🧪 Quiz")
    with st.form("quiz_form", clear_on_submit=False):
        answers = []
        for i,q in enumerate(quiz["questions"]):
            st.markdown(f"**Q{i+1}.** {q.prompt}")
            tts_button(q.prompt)
            ans = st.text_input("Your answer", key=f"ans_{i}")
            answers.append(ans)
            with st.expander("🧮 Calculator"):
                expr = st.text_input("Expression (e.g. 3+4*2)", key=f"calc_{i}")
                if expr:
                    val, steps = calc_with_steps(expr)
                    if val is not None:
                        st.success(f"Result: {val}")
                        st.text_area("Step-by-step", steps, height=120)
                    else:
                        st.error(steps)
            with st.expander("📝 Detailed Solution"):
                st.write(q.explain)
            with st.expander("📝 Notes / rough work"):
                note = st.text_area("Your notes", key=f"note_{i}", value=st.session_state.notes.get(i,""))
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
        progress[user]["records"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "subject": subject,
            "lesson": lesson,
            "difficulty": difficulty,
            "score": correct,
            "total": len(quiz["questions"]),
            "percentage": pct
        })
        save_progress(progress)

        # Feedback
        st.markdown("### 📋 Feedback")
        for i,(qp,u,a,ex,ok) in enumerate(detailed):
            tag = "✅ Correct" if ok else "❌ Incorrect"
            st.markdown(f"**Q{i+1}.** {qp} — {tag}")
            st.markdown(f"- Your answer: `{u}`")
            st.markdown(f"- Correct: `{a}`")
            with st.expander("🧠 Detailed Explanation"):
                st.write(ex)
            st.markdown("---")

        # -------------------------
        # PDF markcard
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
                    x_before = pdf.get_x(); y_before = pdf.get_y()
                    pdf.multi_cell(10, 10, str(idx), border=1, align="C")
                    pdf.set_xy(x_before + 10, y_before)
                    pdf.multi_cell(80, 10, qps[:200], border=1)
                    pdf.set_xy(x_before + 90, y_before)
                    pdf.cell(30, 10, us[:20], border=1)
                    pdf.cell(30, 10, as_[:20], border=1)
                    pdf.cell(40, 10, stat, border=1, ln=1)

                out = pdf.output(dest="S")
                return bytes(out) if isinstance(out, bytearray) else out.encode("latin-1", "ignore")
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
    st.dataframe(df)
    st.bar_chart(df["percentage"])
    st.markdown(f"**Total Quizzes:** {len(df)}  |  **Average Score:** {df['percentage'].mean():.1f}%")
