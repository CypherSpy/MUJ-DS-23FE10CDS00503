"""Code-Mixing Index (CMI), language distribution, and switch points metrics."""

from collections.abc import Sequence

from hinglishlens.schemas import Lang, Token

# All valid language tags
ALL_LANG_TAGS: list[Lang] = ["HI", "EN", "NE", "UNIV", "OTHER"]
MATRIX_LANGS: set[str] = {"HI", "EN"}
INDEPENDENT_LANGS: set[str] = {"UNIV", "NE", "OTHER"}


def compute_lang_distribution(tags: Sequence[str | Token]) -> dict[str, int]:
    """Compute count per language tag across all tokens.

    Ensures all 5 tags ('HI', 'EN', 'NE', 'UNIV', 'OTHER') are present in output dictionary.

    Args:
        tags: Sequence of language tag strings or Token objects.

    Returns:
        Dictionary mapping tag name to frequency count.
    """
    dist = {tag: 0 for tag in ALL_LANG_TAGS}
    for item in tags:
        tag_str = item.lang if isinstance(item, Token) else str(item)
        if tag_str in dist:
            dist[tag_str] += 1
        else:
            dist["OTHER"] += 1
    return dist


def compute_cmi(tags: Sequence[str | Token]) -> float:
    """Compute the Code-Mixing Index (CMI) for a sentence.

    Formula:
        CMI = 0 if n == u, else 100 * (1 - max(w_HI, w_EN) / (n - u))
        where:
        - n: total number of tokens
        - u: count of language-independent tokens (UNIV, NE, OTHER)
        - w_HI, w_EN: counts of HI and EN matrix tokens

    Args:
        tags: Sequence of language tag strings or Token objects.

    Returns:
        CMI score between 0.0 and 100.0 rounded to 2 decimal places.
    """
    n = len(tags)
    if n == 0:
        return 0.0

    tag_strings = [item.lang if isinstance(item, Token) else str(item) for item in tags]

    u = sum(1 for tag in tag_strings if tag in INDEPENDENT_LANGS)
    if n == u:
        return 0.0

    w_hi = sum(1 for tag in tag_strings if tag == "HI")
    w_en = sum(1 for tag in tag_strings if tag == "EN")

    max_w = max(w_hi, w_en)
    matrix_tokens = n - u

    cmi_val = 100.0 * (1.0 - (max_w / matrix_tokens))
    return round(cmi_val, 2)


def compute_switch_points(tags: Sequence[str | Token]) -> int:
    """Compute the number of language switch points between matrix languages (HI <-> EN).

    Language-independent tokens (UNIV, NE, OTHER) are ignored when evaluating switches.

    Args:
        tags: Sequence of language tag strings or Token objects.

    Returns:
        Integer count of switch points.
    """
    matrix_tags = [
        item.lang if isinstance(item, Token) else str(item)
        for item in tags
        if (item.lang if isinstance(item, Token) else str(item)) in MATRIX_LANGS
    ]

    switches = 0
    for i in range(len(matrix_tags) - 1):
        if matrix_tags[i] != matrix_tags[i + 1]:
            switches += 1
    return switches


def compute_metrics(tags: Sequence[str | Token]) -> tuple[float, int, dict[str, int]]:
    """Compute CMI, switch points, and language distribution all at once.

    Args:
        tags: Sequence of language tag strings or Token objects.

    Returns:
        Tuple of (cmi, switch_points, lang_distribution).
    """
    return (
        compute_cmi(tags),
        compute_switch_points(tags),
        compute_lang_distribution(tags),
    )
