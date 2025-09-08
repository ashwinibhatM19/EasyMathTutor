# app.py — Easy Math Tutor (fully fixed)
import os
import json
import random
import subprocess
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# fpdf2 for PDF markcard
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# SymPy for math parsing/explanations
try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Easy Math Tutor Pro", page_icon="🧮", layout="wide")
st.title("🧮 Easy Math Tutor — Pro (Fixed)")
st.caption("Practice • Smart Calculator • AI/Deterministic Explanations • PDF Markcard")

# -------------------------
# persistence
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
# small helpers & dataclasses
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

# sanitize text for PDF output (remove emojis/non-encodable characters)
_RE_HIGH_UNICODE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
def sanitize_for_pdf(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = _RE_HIGH_UNICODE.sub("", s)
    s = "".join(ch for ch in s if ch >= " " or ch == "\n")
    return s

# -------------------------
# Ollama wrapper (optional)
# -------------------------
def call_llama(prompt: str, timeout=15) -> str:
    try:
        proc = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        out = proc.stdout.strip()
        return out if out else proc.stderr.strip() or "AI did not return output."
    except FileNotFoundError:
        return "AI unavailable (Ollama not installed)."
    except subprocess.TimeoutExpired:
        return "AI timed out — try again."
    except Exception as e:
        return f"AI error: {e}"

# -------------------------
# Expression preprocessing & extraction
# -------------------------
def preprocess_expr(expr: Optional[str]) -> str:
    """Normalize user input: convert ^->**, insert * for implicit multiplication, strip commas/spaces."""
    if expr is None:
        return ""
    s = str(expr).strip()
    if s == "":
        return ""
    # replace caret with power
    s = s.replace("^", "**")
    # remove commas (people may paste numbers with commas)
    s = s.replace(",", "")
    # insert * between number and variable or number and '(' e.g. 2x -> 2*x, 2(x -> 2*(x)
    s = re.sub(r"(?P<num>\d)(?P<var>[A-Za-z\(])", r"\g<num>*\g<var>", s)
    # insert * between variable or ')' and '(' or variable/number e.g. x(x -> x*(x), )x -> )*x
    s = re.sub(r"(?P<left>[A-Za-z0-9\)])(?P<right>\()", r"\g<left>*\g<right>", s)
    s = re.sub(r"(?P<left>\))(?P<right>[A-Za-z0-9\(])", r"\g<left>*\g<right>", s)
    # collapse whitespace
    s = re.sub(r"\s+", "", s)
    # tidy accidental sequences
    s = s.replace("* **", "**")
    return s

_EQ_PATTERNS = [
    # capture things like "For 1x^2 + -2x + -3 = 0, sum of roots = ?"
    re.compile(r"([-+*/\w\.\^\*\(\)\s]+)=\s*0"),
    # simple: "Discriminant of 2x^2 - 4x - 30 = ?"
    re.compile(r"of\s+([-+*/\w\.\^\*\(\)\s]+)\s*(?:=|\?)", re.IGNORECASE),
]

def extract_math_from_prompt(prompt: str) -> str:
    """Try to extract the math expression (polynomial or equation) from a question prompt."""
    if not prompt:
        return ""
    p = prompt.strip()
    # try patterns
    for pat in _EQ_PATTERNS:
        m = pat.search(p)
        if m:
            return preprocess_expr(m.group(1))
    # fallback: remove leading words like 'For', 'Compute', etc., and anything after punctuation
    # Try to find the first substring that contains digits and x or usual operators
    m = re.search(r"([-+]?[\dA-Za-z\.\^\*\(\)\/\s\+\-]+)", p)
    if m:
        return preprocess_expr(m.group(1))
    return preprocess_expr(p)

# -------------------------
# SymPy smart eval & deterministic explanations
# -------------------------
if SYMPY_AVAILABLE:
    TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)
else:
    TRANSFORMATIONS = None

# numeric safe evaluator fallback (AST)
import ast, operator as op
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

def smart_eval(expr: str):
    expr_src = preprocess_expr(expr)
    if expr_src == "":
        raise ValueError("Empty expression")
    if SYMPY_AVAILABLE:
        try:
            parsed = parse_expr(expr_src, transformations=TRANSFORMATIONS, evaluate=True)
            simplified = sp.simplify(parsed)
            return simplified
        except Exception as e:
            raise ValueError(f"SymPy parse error: {e}")
    else:
        # fallback: only numeric expressions allowed
        if re.search(r"[A-Za-z]", expr_src):
            raise ValueError("Variables found but SymPy not available")
        try:
            return safe_eval(expr_src)
        except Exception as e:
            raise ValueError(f"Eval error: {e}")

def classroom_explain_from_expr(expr: str) -> str:
    """Produce a classroom-style explanation given a math expression (already preprocessed)."""
    if expr is None or str(expr).strip() == "":
        return "No expression provided."
    if not SYMPY_AVAILABLE:
        return ("SymPy not installed. For deterministic step-by-step explanations, install SymPy:\n"
                "`pip install sympy`.\nProcessed expression: " + preprocess_expr(expr))
    try:
        parsed = parse_expr(expr, transformations=TRANSFORMATIONS, evaluate=True)
    except Exception as e:
        return f"Could not parse expression: {e}"

    symbols = list(parsed.free_symbols)
    # polynomial in single variable? provide polynomial-specific steps
    if len(symbols) == 1:
        x = symbols[0]
        try:
            poly = sp.Poly(parsed, x)
            coeffs = poly.all_coeffs()  # highest -> lowest
            deg = poly.degree()
            if deg == 2 and len(coeffs) == 3:
                a, b, c = coeffs
                disc = sp.simplify(b**2 - 4*a*c)
                # build teacher-style steps
                steps = []
                steps.append("Step 1: Identify coefficients")
                steps.append(f"       a = {a},  b = {b},  c = {c}")
                steps.append("")
                steps.append("Step 2: Recall the discriminant formula")
                steps.append("       Δ = b² - 4ac")
                steps.append("")
                steps.append("Step 3: Substitute values")
                steps.append(f"       Δ = ({b})² - 4*({a})*({c})")
                steps.append(f"       Δ = {sp.simplify(b**2)} - ({4*a*c})")
                steps.append(f"       Δ = {disc}")
                steps.append("")
                if disc.is_real:
                    if disc > 0:
                        steps.append("Step 4: Since Δ > 0, there are two distinct real roots.")
                    elif disc == 0:
                        steps.append("Step 4: Since Δ = 0, there is exactly one real root (repeated).")
                    else:
                        steps.append("Step 4: Since Δ < 0, the roots are complex conjugates.")
                else:
                    steps.append("Step 4: Discriminant is not a real number; roots are complex.")
                # optionally show roots
                try:
                    roots = sp.solve(sp.Eq(parsed, 0), x)
                    steps.append("")
                    steps.append(f"Roots: {roots}")
                except Exception:
                    pass
                return "\n".join(steps)
            else:
                # general polynomial steps: show factorization and numeric roots if reasonable
                lines = []
                lines.append(f"Identified polynomial in {x} of degree {deg}.")
                lines.append(f"Simplified form: {sp.simplify(parsed)}")
                lines.append(f"Coefficients (highest → lowest): {coeffs}")
                try:
                    fact = sp.factor(parsed)
                    if fact != parsed:
                        lines.append(f"Factorization: {fact}")
                except Exception:
                    pass
                if 0 < deg <= 6:
                    try:
                        nr = sp.nroots(parsed)
                        lines.append(f"Numeric roots (approx): {nr}")
                    except Exception:
                        pass
                return "\n".join(map(str, lines))
        except Exception:
            pass

    # fallback generic symbolic explanation
    out = []
    out.append(f"Simplified: {sp.simplify(parsed)}")
    try:
        fact = sp.factor(parsed)
        if fact != parsed:
            out.append(f"Factorized: {fact}")
    except Exception:
        pass
    if symbols:
        try:
            deriv = sp.diff(parsed, symbols[0])
            out.append(f"Derivative wrt {symbols[0]}: {deriv}")
        except Exception:
            pass
    if not symbols:
        try:
            val = sp.N(parsed)
            out.append(f"Numeric value: {val}")
        except Exception:
            pass
    return "\n".join(out)

# convenience wrapper that accepts prompts and extracts the math expression before explaining
def classroom_explain_from_prompt(prompt: str) -> str:
    expr = extract_math_from_prompt(prompt)
    if expr == "":
        # maybe the prompt itself is a simple expression like "2x^2 -4x -30"
        expr = preprocess_expr(prompt)
    return classroom_explain_from_expr(expr)

# -------------------------
# Question generators (same as before)
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
    "School": {
        "Arithmetic": {"Addition/Subtraction": q_arith_addsub},
    },
    "PU": {
        "Algebra": {"Quadratics": q_quadratic},
    },
    "Engineering": {
        "Algebra": {"Quadratics": q_quadratic},
    }
}
DIFFICULTIES = ["Basic","Intermediate","Advanced"]

# -------------------------
# session defaults
# -------------------------
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "descs" not in st.session_state:
    st.session_state.descs = {}
if "notes" not in st.session_state:
    st.session_state.notes = {}

# -------------------------
# sidebar — student + settings
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

    st.markdown("---")
    st.header("⚡ Explanation Mode")
    EXPLAIN_MODE = st.radio("Choose explanation source:", ["Deterministic (SymPy)", "AI (Ollama)"], index=0)
    st.caption("Deterministic is exact; AI gives conversational explanations (Ollama required).")

st.markdown(f"### {level} → {subject} → {lesson}  ·  *{difficulty}*")
st.markdown("Select difficulty then press **Generate Quiz**")

# -------------------------
# Generate Quiz
# -------------------------
gen = CURRICULUM[level][subject][lesson]
if st.button("📝 Generate Quiz"):
    qs = [gen(difficulty) for _ in range(num_q)]
    st.session_state.quiz = {"user": user, "level": level, "subject": subject, "lesson": lesson, "difficulty": difficulty, "questions": qs}
    # generate deterministic concept descriptions using classroom_explain_from_prompt
    descs = {}
    for i,q in enumerate(qs):
        try:
            descs[i] = classroom_explain_from_prompt(q.prompt)
        except Exception:
            descs[i] = q.desc or "Concept summary unavailable."
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
    # height small so it doesn't take much vertical space
    components.html(html, height=40)

# -------------------------
# Render quiz + form (with Calculator, Notepad, Read button)
# -------------------------
if quiz and quiz.get("user")==user:
    st.subheader("🧪 Quiz")
    with st.form("quiz_form", clear_on_submit=False):
        answers = []
        calc_records = {}
        for i,q in enumerate(quiz["questions"]):
            st.markdown(f"**Q{i+1}.** {q.prompt}")
            # concept box (deterministic by default)
            with st.expander("💡 Concept (Auto explanation)"):
                if EXPLAIN_MODE == "Deterministic (SymPy)":
                    # explain by extracting math expression
                    try:
                        st.write(classroom_explain_from_prompt(q.prompt))
                    except Exception as e:
                        st.write(f"Explanation error: {e}")
                else:
                    # AI explain
                    ai_prompt = f"Explain step-by-step like a math teacher how to solve: {q.prompt}"
                    st.write(call_llama(ai_prompt))

            # Read button
            tts_button(q.prompt)

            # answer input
            ans = st.text_input("Your answer", key=f"ans_{i}")
            answers.append(ans)

            # Calculator expander (always present)
            with st.expander("🧮 Calculator"):
                st.write("You can enter algebraic expressions (e.g. `2x^2 - 4x - 30`, `sin(pi/4)`, `(x+1)*(x-2)`).")
                expr = st.text_input("Expression (e.g. 3+4*2)", key=f"calc_{i}")
                if expr:
                    try:
                        val = smart_eval(expr)
                        # pretty display
                        if SYMPY_AVAILABLE:
                            st.success(f"Result: `{sp.simplify(val)}`  (type: {type(val).__name__})")
                        else:
                            st.success(f"Result: {val}")
                        # explanation inside calculator
                        with st.expander("Detailed explanation"):
                            try:
                                expl = classroom_explain_from_expr(preprocess_expr(expr))
                                st.write(expl)
                            except Exception as e:
                                st.write(f"Explanation error: {e}")
                        calc_records[i] = {"expr": expr, "result": str(val)}
                    except Exception as e:
                        st.error(f"Calc error: {e}")
                        calc_records[i] = {"expr": expr, "result": f"Error: {e}"}

            # Notepad (always present)
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

        # save
        progress[user]["records"].append({"level": level, "subject": subject, "lesson": lesson, "difficulty": difficulty, "score": correct, "total": len(quiz["questions"])})
        save_progress(progress)

        # feedback
        st.markdown("### 📋 Feedback")
        for i,(qp,u,a,ex,ok) in enumerate(detailed):
            tag = "✅ Correct" if ok else "❌ Incorrect"
            st.markdown(f"**Q{i+1}.** {qp} — {tag}")
            st.markdown(f"- Your answer: `{u}`")
            st.markdown(f"- Correct: `{a}`")
            with st.expander("🧠 Explanation"):
                if EXPLAIN_MODE == "Deterministic (SymPy)":
                    st.write(classroom_explain_from_prompt(qp))
                else:
                    st.write(call_llama(f"Explain step by step, like a math teacher: {qp}. Final answer: {a}"))
            st.markdown("---")

        # -------------------------
        # PDF markcard builder
        # -------------------------
        def build_pdf_bytes(include_steps: bool=False):
            if not FPDF_AVAILABLE:
                st.error("fpdf2 not installed. Run: pip install fpdf2")
                return None
            try:
                pdf = FPDF(orientation="P", unit="mm", format="A4")
                pdf.set_auto_page_break(True, margin=12)
                pdf.add_page()

                # font
                use_dejavu = os.path.exists(DEJAVU_PATH) and os.access(DEJAVU_PATH, os.R_OK)
                try:
                    if use_dejavu:
                        pdf.add_font("DejaVu", "", DEJAVU_PATH, uni=True)
                        base_font = "DejaVu"
                    else:
                        base_font = "Helvetica"
                except Exception:
                    base_font = "Helvetica"
                    use_dejavu = False

                pdf.set_font(base_font, "", 14)
                pdf.cell(0, 10, sanitize_for_pdf("Easy Math Tutor - Markcard"), ln=True, align="C")
                pdf.ln(4)
                pdf.set_font(base_font, "", 11)
                pdf.cell(0, 6, sanitize_for_pdf(f"Student: {user}"), ln=True)
                pdf.cell(0, 6, sanitize_for_pdf(f"Path: {level}/{subject}/{lesson} ({difficulty})"), ln=True)
                pdf.cell(0, 6, sanitize_for_pdf(f"Score: {correct}/{len(quiz['questions'])} ({pct}%)"), ln=True)
                pdf.ln(6)

                # table header
                colw = {"no": 10, "q": 95, "ya": 30, "ca": 30, "st": 25}
                pdf.set_font(base_font, "B", 11)
                pdf.set_fill_color(200, 220, 255)
                pdf.cell(colw["no"], 10, "No", border=1, align="C", fill=True)
                pdf.cell(colw["q"], 10, "Question", border=1, align="C", fill=True)
                pdf.cell(colw["ya"], 10, "Your Ans", border=1, align="C", fill=True)
                pdf.cell(colw["ca"], 10, "Correct", border=1, align="C", fill=True)
                pdf.cell(colw["st"], 10, "Status", border=1, align="C", ln=1, fill=True)

                pdf.set_font(base_font, "", 10)
                for idx, (qp, u, a, ex, ok) in enumerate(detailed, start=1):
                    qps = sanitize_for_pdf(qp)
                    us = sanitize_for_pdf(u)
                    as_ = sanitize_for_pdf(a)
                    stat = "Correct" if ok else "Wrong"

                    x0 = pdf.get_x()
                    y0 = pdf.get_y()

                    pdf.multi_cell(colw["no"], 8, str(idx), border=1, align="C")
                    pdf.set_xy(x0 + colw["no"], y0)
                    pdf.multi_cell(colw["q"], 8, qps, border=1)
                    y_row_end = pdf.get_y()
                    pdf.set_xy(x0 + colw["no"] + colw["q"], y0)
                    pdf.multi_cell(colw["ya"], 8, us, border=1, align="C")
                    pdf.set_xy(x0 + colw["no"] + colw["q"] + colw["ya"], y0)
                    pdf.multi_cell(colw["ca"], 8, as_, border=1, align="C")
                    pdf.set_xy(x0 + colw["no"] + colw["q"] + colw["ya"] + colw["ca"], y0)
                    pdf.multi_cell(colw["st"], 8, stat, border=1, align="C")
                    pdf.set_xy(10, max(y_row_end, pdf.get_y()))

                    # optionally include the step-by-step explanation on following page (disabled by default)
                    if include_steps:
                        pdf.ln(2)
                        pdf.set_font(base_font, "B", 11)
                        pdf.cell(0, 6, sanitize_for_pdf(f"Solution Q{idx}:"), ln=True)
                        pdf.set_font(base_font, "", 10)
                        expl = classroom_explain_from_prompt(qp) if EXPLAIN_MODE == "Deterministic (SymPy)" else call_llama(f"Explain step by step: {qp}")
                        # write explanation wrapped
                        pdf.multi_cell(0, 6, sanitize_for_pdf(expl))
                        pdf.add_page()

                out = pdf.output(dest="S")
                if isinstance(out, str):
                    return out.encode("latin-1", "ignore")
                else:
                    return out
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                return None

        pdf_bytes = build_pdf_bytes(include_steps=False)
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
    df["percent"] = 100 * df["score"] / df["total"]
    st.metric("Attempts", len(df))
    st.write(df.sort_values(by="percent", ascending=False).reset_index(drop=True))
else:
    st.info("No attempts yet — generate a quiz and submit answers to see analytics.")
