"""Curated stylistic keyword set used to split TMDb keywords into
'style' (mood/tone/genre-tropes) vs 'plot' (subject matter)."""

from __future__ import annotations

STYLISTIC_DICT: frozenset[str] = frozenset({
    # noir family
    "neo-noir", "film noir", "noir",
    # horror sub-genres
    "slasher", "psychological horror", "supernatural horror",
    "body horror", "cosmic horror", "gothic",
    # comedy sub-genres
    "mockumentary", "satire", "parody", "dark comedy",
    "screwball comedy", "romantic comedy",
    # thriller sub-genres
    "psychological thriller",
    # cinematic style markers
    "atmospheric", "surrealism", "experimental", "art house",
    "indie film", "cult classic",
    # setting / world style
    "dystopian future", "post-apocalyptic", "cyberpunk", "steampunk",
    # western sub-genres
    "neo-western", "spaghetti western",
    # narrative form
    "anthology film", "non-linear narrative", "unreliable narrator",
    "ensemble cast", "one-shot", "epistolary",
    # production style
    "silent film", "black and white", "musical",
    "stop motion", "claymation", "anime", "rotoscoping",
    "found footage",
    # genre tropes
    "buddy cop", "courtroom drama", "heist film",
    "war epic", "coming-of-age", "fish out of water",
})

MAX_KEYWORDS_PER_BUCKET = 8


def split_keywords(keywords: list[str]) -> tuple[list[str], list[str]]:
    """Split a TMDb keyword list into (style, plot) by STYLISTIC_DICT.

    Match is case-insensitive on keyword name. Original case is
    preserved in output. Order is preserved. Each output array is
    capped at MAX_KEYWORDS_PER_BUCKET entries.

    Per spec §5.5: if the dict yields zero hits, the spec amendment
    drops the dual presentation entirely; this function still returns
    `(style=[], plot=keywords_capped)` and the caller decides how
    to render.
    """
    style: list[str] = []
    plot: list[str] = []
    lower_dict = STYLISTIC_DICT  # already lowercase frozenset
    for kw in keywords:
        if kw.lower() in lower_dict:
            style.append(kw)
        else:
            plot.append(kw)
    return style[:MAX_KEYWORDS_PER_BUCKET], plot[:MAX_KEYWORDS_PER_BUCKET]
