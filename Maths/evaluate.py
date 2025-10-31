from typing import Tuple
import re


ALLOWED_CHARS = set("0123456789+-*/() ")


def _normalize(expr: str) -> str:
    """Normalize common handwriting/Unicode variants and balance parentheses.

    - Map common Unicode variants (×, ÷, Unicode dashes) to ASCII.
    - Drop any characters outside a safe superset before final allowlist.
    - Remove unmatched ')' and append missing ')' to balance.
    - Trim trailing operators.
    """
    if not expr:
        return expr

    # Map common variants
    trans_pairs = [
        ("×", "*"), ("x", "*"), ("X", "*"), ("·", "*"),
        ("÷", "/"),
        ("−", "-"), ("–", "-"), ("—", "-"), ("‑", "-"),
        ("“", '"'), ("”", '"'), (" ", " "), (" ", " "), (" ", " "),
    ]
    for k, v in trans_pairs:
        expr = expr.replace(k, v)

    # Keep only allowed superset chars
    superset = set("0123456789+-*/() ")
    filtered = []
    for ch in expr:
        if ch in superset:
            filtered.append(ch)
        # ignore all others
    expr = "".join(filtered)

    # Remove unmatched ')' while scanning and count '(' to close later
    balanced = []
    depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
            balanced.append(ch)
        elif ch == ')':
            if depth > 0:
                depth -= 1
                balanced.append(ch)
            else:
                # skip unmatched ')'
                continue
        else:
            balanced.append(ch)

    # Append missing ')' to close any remaining '('
    balanced.extend(')' for _ in range(depth))

    expr = "".join(balanced)

    # Collapse multiple spaces
    expr = re.sub(r"\s+", " ", expr).strip()

    # Join digits split by spaces (e.g., '1 0' -> '10')
    # This can happen if segmentation inserts space between adjacent digit glyphs
    expr = re.sub(r"(?<=\d)\s+(?=\d)", "", expr)

    # Optional: insert implicit multiplication for cases the predictor missed
    # e.g., '2(' -> '2*(' and ')3' -> ')*3'
    expr = re.sub(r"(\d)\(", r"\1*(", expr)
    expr = re.sub(r"\)(\d)", r")*\1", expr)

    # Collapse runs of operators introduced by over-segmentation
    # Example: '85+++35' -> '85+35'
    def _collapse_ops(m: re.Match) -> str:
        s = m.group(0)
        return s[-1]  # keep the last operator

    expr = re.sub(r"[+\-*/]{2,}", _collapse_ops, expr)

    # Trim trailing operators
    while expr and expr[-1] in "+-*/":
        expr = expr[:-1].rstrip()

    return expr


def safe_eval(expression: str) -> Tuple[str, str]:
    """
    Safely evaluate an arithmetic expression using SymPy.
    Returns (pretty_expression, result_string).
    """
    expr = expression.strip()
    if not expr:
        return "", ""

    # Normalize and basic sanitization
    expr = _normalize(expr)
    # Be tolerant of empty or operator-only results
    if not expr:
        return "", ""
    if all(ch in "+-*/ " for ch in expr):
        return expr.strip(), ""
    if any(ch not in ALLOWED_CHARS for ch in expr):
        # Filter unknowns instead of raising
        expr = "".join(ch for ch in expr if ch in ALLOWED_CHARS)
        if not expr or all(ch in "+-*/ " for ch in expr):
            return expr.strip(), ""

    try:
        # Lazy import to avoid slowing GUI startup
        import sympy as sp
        sy = sp.sympify(expr, evaluate=True)

        # Try exact simplification first
        simplified = sp.simplify(sy)
        if simplified.is_Number:
            if simplified.is_Integer:
                result_str = str(int(simplified))
            elif isinstance(simplified, sp.Rational):
                # Proper fraction or reducible
                if simplified.q == 1:
                    result_str = str(int(simplified))
                else:
                    result_str = str(simplified)  # like 7/2
            else:  # Float
                num = float(simplified.evalf())
                result_str = ("{:.6f}".format(num)).rstrip("0").rstrip(".")
        else:
            # Fall back to numerical evaluation
            num = float(sp.N(sy))
            result_str = ("{:.6f}".format(num)).rstrip("0").rstrip(".")
    except Exception:
        # Return sanitized expression with empty result on failure; avoids GUI popups
        return expr, ""

    return expr, result_str

