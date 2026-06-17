def _dedup_merge_texts(texts: list) -> str:
    """Merge consecutive entry texts, removing carryover overlap at boundaries.

    Auto-captions produce entries where entry N+1 repeats the last ~5-15 words
    of entry N before continuing with new content. This function detects and
    removes those boundary overlaps so the final text reads as natural speech.
    """
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]

    result = texts[0]

    for i in range(1, len(texts)):
        current = texts[i].strip()
        if not current:
            continue

        # Get last ~20 words of accumulated text
        result_words = result.split()
        suffix = " ".join(result_words[-20:]).lower()

        # Find the longest overlap between suffix of result and prefix of current
        current_lower = current.lower()
        best_overlap_len = 0

        # Try overlap lengths from longest to shortest
        for ov in range(min(20, len(result_words), len(current.split())), 2, -1):
            # Candidate suffix from result
            cand = " ".join(result_words[-ov:]).lower()
            # Check if current text starts with this suffix
            if current_lower.startswith(cand):
                best_overlap_len = ov
                break

        if best_overlap_len > 0:
            # Strip the overlapped prefix from current text
            overlap_words_count = best_overlap_len
            overlap_str = " ".join(result_words[-overlap_words_count:])
            # The current text may start with slightly different casing/punctuation
            # Find where the overlap ends in the current text
            trimmed = current[len(overlap_str):].strip()
            if trimmed:
                result += " " + trimmed
            # If trimmed is empty, the current entry added nothing new — skip it
        else:
            result += " " + current

    return result
