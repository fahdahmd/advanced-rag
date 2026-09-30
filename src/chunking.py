def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[tuple[int, str]]:
    """Slide a window over the text. Returns (start_offset, chunk) pairs."""
    chunks = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # step back to the last space/newline so we don't cut a word in half
            cut = max(text.rfind(" ", start + size - 100, end), text.rfind("\n", start + size - 100, end))
            if cut > start:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, chunk))
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # overlap keeps facts from being cut in two
    return chunks