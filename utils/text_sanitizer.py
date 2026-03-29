# -*- coding: utf-8 -*-
"""文本标签清洗与双通道输出。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.constants import (
    MINIMAX_EXPRESSIVE_MODELS,
    MINIMAX_EXPRESSIVE_TAGS,
    PLUGIN_DIR,
)
from ..utils.extract import CodeAndLinkExtractor

logger = logging.getLogger(__name__)

DEFAULT_MEME_JSON_PATH = PLUGIN_DIR.parent.parent / "memes_data" / "memes_data.json"
DEFAULT_MEME_DIR = PLUGIN_DIR.parent.parent / "memes_data" / "memes"

_VOICE_TAG_PATTERN = "|".join(re.escape(tag) for tag in MINIMAX_EXPRESSIVE_TAGS)
_VOICE_TAG_RE = re.compile(rf"\(({_VOICE_TAG_PATTERN})\)", re.IGNORECASE)
_PAUSE_TAG_RE = re.compile(r"<#\s*(\d{1,2}(?:\.\d{1,2})?)\s*#>")
_STRICT_MEME_TAG_RE = re.compile(r"&&([^&\n]+)&&")
_BRACKET_MEME_TAG_RE = re.compile(r"\[([^\[\]\n]+)\](?!\()")
_PAREN_MEME_TAG_RE = re.compile(r"\(([^()\n]+)\)")


@dataclass
class PreparedSpeechText:
    """清洗后的双通道文本。"""

    source_text: str
    tts_text: str
    display_text: str
    detected_emotion: Optional[str] = None
    links: List[str] = field(default_factory=list)
    codes: List[str] = field(default_factory=list)
    matched_tags: Dict[str, List[str]] = field(default_factory=dict)


class SpeechTextSanitizer:
    """负责把展示文本和 TTS 文本拆开。"""

    def __init__(
        self,
        *,
        marker_processor,
        extractor: Optional[CodeAndLinkExtractor] = None,
        meme_json_path: Optional[Path] = None,
        meme_dir: Optional[Path] = None,
    ):
        self.marker_processor = marker_processor
        self.extractor = extractor or CodeAndLinkExtractor()
        self.meme_json_path = Path(meme_json_path or DEFAULT_MEME_JSON_PATH)
        self.meme_dir = Path(meme_dir or DEFAULT_MEME_DIR)
        self._meme_tags_cache: Optional[set[str]] = None
        self._meme_tags_cache_mtime: Optional[float] = None

    def prepare(self, text: str, *, provider: str, model: str) -> PreparedSpeechText:
        raw_text = self.marker_processor.normalize_text(text or "")
        cleaned_text, detected_emotion = self.marker_processor.strip_head_many(raw_text)
        cleaned_text = self.marker_processor.strip_all_visible_markers(cleaned_text)

        meme_cleaned_text, meme_tags = self._strip_meme_tags(cleaned_text)
        display_base, pause_tags = self._handle_pause_tags(meme_cleaned_text, keep=False)
        display_base, voice_tags = self._handle_voice_tags(display_base, keep=False)

        keep_minimax_controls = self._should_keep_minimax_controls(provider)
        keep_voice_tags = keep_minimax_controls and self._supports_expressive_tags(model)
        tts_base, _ = self._handle_pause_tags(meme_cleaned_text, keep=keep_minimax_controls)
        tts_base, _ = self._handle_voice_tags(tts_base, keep=keep_voice_tags)

        display_processed = self.extractor.process_text(display_base)
        tts_processed = self.extractor.process_text(
            tts_base,
            preserve_linebreaks=keep_minimax_controls,
        )

        display_text = self._cleanup_visible_text(display_processed.clean_text, keep_newlines=True)
        tts_text = self._cleanup_visible_text(
            tts_processed.speak_text,
            keep_newlines=keep_minimax_controls,
        )

        matched_tags = {
            "meme": meme_tags,
            "pause": pause_tags,
            "minimax_voice": voice_tags,
        }
        matched_tags = {key: value for key, value in matched_tags.items() if value}

        return PreparedSpeechText(
            source_text=text or "",
            tts_text=tts_text,
            display_text=display_text,
            detected_emotion=detected_emotion,
            links=list(display_processed.links),
            codes=list(display_processed.codes),
            matched_tags=matched_tags,
        )

    def _should_keep_minimax_controls(self, provider: str) -> bool:
        return str(provider or "").strip().lower() == "minimax"

    def _supports_expressive_tags(self, model: str) -> bool:
        return str(model or "").strip().lower() in MINIMAX_EXPRESSIVE_MODELS

    def _load_meme_tags(self) -> set[str]:
        json_path = self.meme_json_path
        dir_path = self.meme_dir

        current_mtime = None
        if json_path.exists():
            current_mtime = json_path.stat().st_mtime
        elif dir_path.exists():
            current_mtime = dir_path.stat().st_mtime

        if self._meme_tags_cache is not None and current_mtime == self._meme_tags_cache_mtime:
            return self._meme_tags_cache

        tags: set[str] = set()

        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for key in data:
                        if isinstance(key, str) and key.strip():
                            tags.add(key.strip().lower())
            except Exception:
                logger.warning("load meme tags from json failed: %s", json_path, exc_info=True)

        if not tags and dir_path.exists():
            try:
                for path in dir_path.iterdir():
                    if path.is_dir() and path.name.strip():
                        tags.add(path.name.strip().lower())
            except Exception:
                logger.warning("load meme tags from dir failed: %s", dir_path, exc_info=True)

        self._meme_tags_cache = tags
        self._meme_tags_cache_mtime = current_mtime
        return tags

    def _strip_meme_tags(self, text: str) -> Tuple[str, List[str]]:
        valid_tags = self._load_meme_tags()
        if not text or not valid_tags:
            return text, []

        matched: List[str] = []

        def _replace_if_valid(match: re.Match, *, allow_alt: bool = True) -> str:
            tag = (match.group(1) or "").strip().lower()
            if not tag:
                return match.group(0)
            if tag not in valid_tags:
                return match.group(0)
            if not allow_alt:
                matched.append(tag)
                return ""
            matched.append(tag)
            return ""

        text = _STRICT_MEME_TAG_RE.sub(lambda m: _replace_if_valid(m, allow_alt=False), text)
        text = _BRACKET_MEME_TAG_RE.sub(_replace_if_valid, text)
        text = _PAREN_MEME_TAG_RE.sub(_replace_if_valid, text)

        return text, self._dedupe_preserve_order(matched)

    def _handle_pause_tags(self, text: str, *, keep: bool) -> Tuple[str, List[str]]:
        matched: List[str] = []

        def _replace(match: re.Match) -> str:
            matched.append(match.group(0))
            return match.group(0) if keep else ""

        return _PAUSE_TAG_RE.sub(_replace, text), self._dedupe_preserve_order(matched)

    def _handle_voice_tags(self, text: str, *, keep: bool) -> Tuple[str, List[str]]:
        matched: List[str] = []

        def _replace(match: re.Match) -> str:
            matched.append(f"({(match.group(1) or '').lower()})")
            return match.group(0) if keep else ""

        return _VOICE_TAG_RE.sub(_replace, text), self._dedupe_preserve_order(matched)

    def _cleanup_visible_text(self, text: str, *, keep_newlines: bool) -> str:
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)

        if keep_newlines:
            text = re.sub(r"\n{3,}", "\n\n", text)
        else:
            text = re.sub(r"\s+", " ", text)

        return text.strip()

    @staticmethod
    def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered
