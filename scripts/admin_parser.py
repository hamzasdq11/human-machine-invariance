"""Parser and prompt frames copied verbatim from the 22 Aug administration cell."""
_PUNCT = str.maketrans("", "", ".,;:!?()[]{}\"'`")
def _norm(s): return " ".join(str(s).lower().translate(_PUNCT).split())

def parse_free_response(text, options):
    """Match a stated answer back to an option. Ambiguity is missing, not a guess."""
    if not text: return None
    t = _norm(text)
    if not t: return None
    opts = [_norm(o) for o in options]
    for i, o in enumerate(opts):
        if t == o: return i + 1
    toks = t.split(); hits = []
    for i, o in enumerate(opts):
        ow = o.split()
        if len(ow) == 1:
            if o in toks: hits.append(i + 1)
        else:
            for j in range(len(toks) - len(ow) + 1):
                if toks[j:j+len(ow)] == ow: hits.append(i + 1); break
    if len(set(hits)) == 1: return hits[0]
    if len(set(hits)) > 1: return None
    hits = [i + 1 for i, o in enumerate(opts) if o and t.startswith(o)]
    return hits[0] if len(hits) == 1 else None

FREE_FRAMES = {
 "v1": ("Answer the following question.\n\n{stem}\n\nThe possible answers are:\n{options}\n\n"
        "Reply with the answer itself, exactly as written above, and nothing else."),
 "v2": ("Below is a question with a list of possible answers.\n\n{stem}\n\n{options}\n\n"
        "State the correct answer exactly as it appears in the list. Nothing else."),
 "v3": ("Question:\n{stem}\n\nPossible answers:\n{options}\n\n"
        "Which is correct? Reply with the answer text only."),
 "v4": ("You are taking a reasoning test.\n\n{stem}\n\nChoose from:\n{options}\n\n"
        "Write the correct answer exactly as listed, and nothing more."),
 "v5": "{stem}\n\n{options}\n\nAnswer (copy the correct option exactly):",
}

