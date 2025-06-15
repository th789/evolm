from math_verify import parse, verify


def latex_equiv(latex_expr_1, latex_expr_2):
    """Check if two expressions are equivalent using LaTeX."""
    try:
        return verify(parse(latex_expr_1), parse(latex_expr_2))
    except Exception as e:
        print(f"Error in `math_verify_toolkit.latex_equiv` when verifying {latex_expr_1} and {latex_expr_2}: {e}")
        return False
