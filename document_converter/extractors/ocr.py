"""Local OCR helper (spec 6.6). Optional dependency: tesseract must be on PATH."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


class OCRUnavailableError(RuntimeError):
    pass


@dataclass
class OCRLine:
    """One physical text line recovered from `image_to_data`, with its real
    bounding box (image pixel space, top-left origin) and mean per-word
    confidence (0-100; -1 when tesseract reported none). This is genuine
    layout/quality evidence read back from the OCR engine itself -- never
    synthesized -- so callers can use line height as real (if noisy)
    typographic evidence and confidence as a real signal, instead of
    inventing both from blank-line isolation alone."""
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float


def check_ocr_available() -> None:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception as e:
        raise OCRUnavailableError(
            "OCR was requested but the local Tesseract binary is not available on this "
            "machine. Install it (e.g. `sudo apt install tesseract-ocr` on Debian/Ubuntu, "
            "`brew install tesseract` on macOS) so `tesseract` is on PATH, or disable OCR "
            "with --no-ocr / ocr mode 'off'. All OCR runs locally; nothing is sent to a "
            "remote service."
        ) from e


def ocr_image(image, language: str = "eng") -> str:
    """OCR a PIL image locally with pytesseract. Raises OCRUnavailableError if unavailable."""
    check_ocr_available()
    import pytesseract
    return pytesseract.image_to_string(image, lang=language)


def ocr_image_lines(image, language: str = "eng") -> List[OCRLine]:
    """OCR a PIL image locally with pytesseract's word-level `image_to_data`,
    grouping words back into physical lines (by tesseract's own block/
    paragraph/line numbering, which already reflects reading order) while
    keeping each line's real bounding box and mean word confidence. This is
    the layout-aware counterpart to `ocr_image`: `image_to_string` throws
    away exactly the two pieces of genuine evidence (line height, per-word
    confidence) that a heading classifier needs so it never has to fall back
    to manufacturing them from blank-line isolation alone."""
    check_ocr_available()
    import pytesseract

    data = pytesseract.image_to_data(image, lang=language, output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))
    grouped: dict = {}
    order: List[tuple] = []
    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in grouped:
            grouped[key] = {"words": [], "confs": [], "x0": None, "y0": None, "x1": None, "y1": None}
            order.append(key)
        rec = grouped[key]
        rec["words"].append(word)
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf >= 0:
            rec["confs"].append(conf)
        x, y = float(data["left"][i]), float(data["top"][i])
        x1, y1 = x + float(data["width"][i]), y + float(data["height"][i])
        rec["x0"] = x if rec["x0"] is None else min(rec["x0"], x)
        rec["y0"] = y if rec["y0"] is None else min(rec["y0"], y)
        rec["x1"] = x1 if rec["x1"] is None else max(rec["x1"], x1)
        rec["y1"] = y1 if rec["y1"] is None else max(rec["y1"], y1)

    out: List[OCRLine] = []
    for key in order:
        rec = grouped[key]
        conf = (sum(rec["confs"]) / len(rec["confs"])) if rec["confs"] else -1.0
        out.append(OCRLine(
            text=" ".join(rec["words"]), conf=conf,
            x0=rec["x0"], y0=rec["y0"], x1=rec["x1"], y1=rec["y1"],
        ))
    return out
