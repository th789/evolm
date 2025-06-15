import re
from typing import Tuple


def is_number(s) -> Tuple[bool, str]:
    try:
        res = float(s)
        return True, str(res)
    except:
        pass

    try:
        import unicodedata

        res = unicodedata.numeric(s)
        return True, str(res)
    except:
        pass

    return False, None


def avg(lst):
    assert isinstance(lst, list) and len(lst) > 0
    assert all([isinstance(x, (int, float, bool)) for x in lst])
    return sum(lst) / len(lst)


complex_number_pattern = re.compile(
    r"""
    # Complex number indicators
    \\mathbb\{C\}|        # Complex number set ℂ
    \\i\b|                # Complex i
    \bi\b|                # Standalone i
    \\text\{i\}|          # Text i
    \\mathrm\{i\}|        # Roman i
    \\imath\b|            # Alternative i notation

    # Matrix operations
    \\det|                # Determinant
    \\operatorname\{tr\}| # Trace
    \\operatorname\{rank\}| # Rank
    \\text\{rank\}|
    \\arg\{|              # Complex argument
    \\Re\{|               # Real part
    \\Im\{|               # Imaginary part
    \\operatorname\{Re\}| # Real part alternate
    \\operatorname\{Im\}| # Imaginary part alternate
    \\text\{Re\}|         # Real part text
    \\text\{Im\}          # Imaginary part text
""",
    re.VERBOSE,
)


def should_treat_as_complex(latex_str: str) -> bool:
    """
    Returns True if the latex string likely contains complex numbers, matrices, or vectors.
    """

    return bool(complex_number_pattern.search(latex_str))
