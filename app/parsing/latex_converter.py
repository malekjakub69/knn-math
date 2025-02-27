"""
Modul pro převod rozpoznaných symbolů do LaTeX formátu
"""


def convert_to_latex(symbols):
    """
    Převod posloupnosti rozpoznaných symbolů do LaTeX formátu

    Args:
        symbols: Seznam rozpoznaných symbolů

    Returns:
        LaTeX kód odpovídající rozpoznaným symbolům
    """
    # Mapování symbolů na jejich LaTeX reprezentaci
    latex_map = {
        "+": "+",
        "-": "-",
        "*": "\\times",
        "/": "\\div",
        "=": "=",
        "<": "<",
        ">": ">",
        "(": "(",
        ")": ")",
        "[": "[",
        "]": "]",
        "{": "\\{",
        "}": "\\}",
        "sqrt": "\\sqrt",
        "sum": "\\sum",
        "int": "\\int",
        "inf": "\\infty",
        "alpha": "\\alpha",
        "beta": "\\beta",
        "gamma": "\\gamma",
        "delta": "\\delta",
        "pi": "\\pi",
        "theta": "\\theta",
        "sigma": "\\sigma",
        "omega": "\\omega",
    }

    # Převod symbolů na LaTeX
    latex_code = ""
    for symbol in symbols:
        if symbol in latex_map:
            latex_code += latex_map[symbol] + " "
        else:
            # Pokud symbol není ve slovníku, použijeme jej přímo
            latex_code += symbol + " "

    # Zabalení do matematického prostředí
    return "$" + latex_code.strip() + "$"
