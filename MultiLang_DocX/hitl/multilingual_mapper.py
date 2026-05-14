"""
Multilingual Key Mapping (Phase 4.3 — Minor Step)
==================================================
Ensures that entities extracted in native scripts (Tamil, Hindi, Arabic, etc.)
are properly mapped with:
  - original_script: raw value as extracted by the VLM
  - transliterated:  IAST/Latin transliteration of the script
  - english:         best-effort English translation / transcription

Design:
  - Uses `indic-transliteration` for Indic scripts (Devanagari, Tamil, Telugu, etc.)
  - Uses `arabic-reshaper` + `python-bidi` awareness for Arabic
  - Gracefully degrades if optional packages not installed
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Optional dependency imports ───────────────────────────────────────────────

_INDIC_AVAILABLE = False
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as indic_transliterate
    _INDIC_AVAILABLE = True
    logger.info("indic-transliteration available.")
except ImportError:
    logger.warning(
        "indic-transliteration not installed. "
        "Install with: pip install indic-transliteration"
    )

_TRANSLITERATE_AVAILABLE = False
try:
    from transliterate import translit, get_available_language_codes  # type: ignore
    _TRANSLITERATE_AVAILABLE = True
    logger.info("transliterate library available.")
except ImportError:
    logger.warning(
        "transliterate not installed. "
        "Install with: pip install transliterate"
    )


# ── Unicode block → script name mapping ──────────────────────────────────────

_UNICODE_BLOCKS: List[Tuple[int, int, str]] = [
    (0x0900, 0x097F, "devanagari"),    # Hindi, Sanskrit, Marathi
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),      # Punjabi
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "odia"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
    (0x0600, 0x06FF, "arabic"),
    (0x0750, 0x077F, "arabic"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0400, 0x04FF, "cyrillic"),
    (0x4E00, 0x9FFF, "cjk"),           # Chinese/Japanese/Korean
    (0x3040, 0x309F, "hiragana"),
    (0x30A0, 0x30FF, "katakana"),
    (0xAC00, 0xD7AF, "hangul"),
    (0x0E00, 0x0E7F, "thai"),
    (0x0000, 0x007F, "latin"),         # ASCII
    (0x0080, 0x024F, "latin"),         # Extended Latin
]


def detect_script(text: str) -> str:
    """
    Detect the dominant Unicode script in `text`.
    Returns one of: 'latin', 'devanagari', 'tamil', 'arabic', 'bengali',
    'telugu', 'kannada', 'malayalam', 'gujarati', 'gurmukhi', 'odia',
    'cyrillic', 'cjk', 'hiragana', 'katakana', 'hangul', 'thai', 'unknown'.
    """
    if not text or not text.strip():
        return "unknown"

    script_counts: Dict[str, int] = {}
    for char in text:
        cp = ord(char)
        for start, end, script in _UNICODE_BLOCKS:
            if start <= cp <= end:
                script_counts[script] = script_counts.get(script, 0) + 1
                break

    if not script_counts:
        return "unknown"

    dominant = max(script_counts, key=script_counts.__getitem__)
    # Only flag as non-latin if non-latin chars dominate
    latin_count = script_counts.get("latin", 0)
    non_latin_count = sum(v for k, v in script_counts.items() if k != "latin")
    if non_latin_count > latin_count:
        return dominant
    return "latin"


def _transliterate_indic(text: str, script: str) -> Optional[str]:
    """Attempt Indic transliteration using indic-transliteration."""
    if not _INDIC_AVAILABLE:
        return None

    script_map = {
        "devanagari": sanscript.DEVANAGARI,
        "bengali":    sanscript.BENGALI,
        "gujarati":   sanscript.GUJARATI,
        "gurmukhi":   sanscript.GURMUKHI,
        "telugu":     sanscript.TELUGU,
        "kannada":    sanscript.KANNADA,
        "malayalam":  sanscript.MALAYALAM,
        "tamil":      sanscript.TAMIL,
        "odia":       sanscript.ORIYA,
    }

    src_scheme = script_map.get(script)
    if src_scheme is None:
        return None

    try:
        return indic_transliterate(text, src_scheme, sanscript.IAST)
    except Exception as e:
        logger.debug("Indic transliteration failed for '%s': %s", text[:30], e)
        return None


def _transliterate_generic(text: str, script: str) -> Optional[str]:
    """Attempt generic transliteration using the `transliterate` library."""
    if not _TRANSLITERATE_AVAILABLE:
        return None

    lang_map = {
        "cyrillic": "ru",
        "greek":    "el",
        "armenian": "hy",
        "georgian": "ka",
    }
    lang_code = lang_map.get(script)
    if lang_code is None:
        return None

    try:
        return translit(text, lang_code, reversed=True)
    except Exception as e:
        logger.debug("Generic transliteration failed for '%s': %s", text[:30], e)
        return None


def transliterate_text(text: str, script: Optional[str] = None) -> Optional[str]:
    """
    Best-effort transliteration of `text` to Latin/IAST.
    Returns None if no transliteration is possible.
    """
    if not text or not text.strip():
        return None

    detected = script or detect_script(text)

    if detected == "latin":
        return text  # Already Latin; no conversion needed

    # Try Indic first
    result = _transliterate_indic(text, detected)
    if result:
        return result

    # Try generic
    result = _transliterate_generic(text, detected)
    if result:
        return result

    # Minimal fallback: Unicode NFKD decomposition strips diacritics
    try:
        normalized = unicodedata.normalize("NFKD", text)
        ascii_approx = normalized.encode("ascii", "ignore").decode("ascii")
        if ascii_approx.strip():
            return ascii_approx
    except Exception:
        pass

    return None


# ── Entity mapper ─────────────────────────────────────────────────────────────

class MultilingualMapper:
    """
    Phase 4.3: Walk the aggregated JSON and annotate every non-Latin string
    with its transliterated and (if possible) English equivalents.

    Output for each native-script value:
        {
            "field_path": "people[0].name",
            "original_script": "குமார்",
            "script_detected": "tamil",
            "transliterated": "Kumār",
            "english": "Kumar",       ← same as transliterated for Indic
        }
    """

    def map_entity(self, value: str, field_path: str = "") -> Dict[str, str]:
        """Map a single string value to a multilingual annotation dict."""
        script = detect_script(value)
        transliterated = transliterate_text(value, script)
        return {
            "field_path": field_path,
            "original_script": value,
            "script_detected": script,
            "transliterated": transliterated or value,
            "english": transliterated or value,  # Best effort; extend with MT if needed
        }

    def process_document(
        self,
        aggregated_json: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Walk the entire aggregated document JSON and collect multilingual mappings
        for every non-Latin string value.

        Returns a list of mapping dicts (one per detected non-Latin value).
        """
        mappings: List[Dict[str, str]] = []
        self._walk(aggregated_json, "", mappings)
        return mappings

    def enrich_document(
        self,
        aggregated_json: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """
        Returns (enriched_json, mappings) where enriched_json has in-place
        transliterated values added under a `_transliterated` sibling key,
        and mappings is the full multilingual annotation list.
        """
        import copy
        enriched = copy.deepcopy(aggregated_json)
        mappings = self.process_document(aggregated_json)

        # Build path → transliteration lookup
        path_map = {m["field_path"]: m for m in mappings}

        # Inject transliterations into the enriched copy
        self._inject(enriched, "", path_map)
        return enriched, mappings

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _walk(
        self,
        obj: Any,
        prefix: str,
        mappings: List[Dict[str, str]],
    ) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else k
                self._walk(v, path, mappings)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                path = f"{prefix}[{i}]"
                self._walk(item, path, mappings)
        elif isinstance(obj, str) and obj.strip():
            script = detect_script(obj)
            if script not in ("latin", "unknown"):
                mappings.append(self.map_entity(obj, prefix))

    def _inject(
        self,
        obj: Any,
        prefix: str,
        path_map: Dict[str, Dict[str, str]],
    ) -> None:
        """Inject `_en` sibling keys for non-Latin values in a dict."""
        if isinstance(obj, dict):
            keys_to_add: Dict[str, str] = {}
            for k, v in list(obj.items()):
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, str) and path in path_map:
                    keys_to_add[f"{k}_en"] = path_map[path]["english"]
                else:
                    self._inject(v, path, path_map)
            obj.update(keys_to_add)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                path = f"{prefix}[{i}]"
                self._inject(item, path, path_map)
