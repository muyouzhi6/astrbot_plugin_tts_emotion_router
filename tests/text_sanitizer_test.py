import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from astrbot_plugin_tts_emotion_router.utils.extract import CodeAndLinkExtractor
from astrbot_plugin_tts_emotion_router.utils.text_sanitizer import SpeechTextSanitizer


class _MarkerProcessor:
    def normalize_text(self, text):
        return text

    def strip_head_many(self, text):
        return text, None

    def strip_all_visible_markers(self, text):
        return text


class SpeechTextSanitizerTests(unittest.TestCase):
    def _sanitizer(self):
        return SpeechTextSanitizer(
            marker_processor=_MarkerProcessor(),
            extractor=CodeAndLinkExtractor(),
        )

    def test_simple_filename_inline_code_is_not_appended_as_code_reference(self):
        prepared = self._sanitizer().prepare(
            "好的，那发一个 `config.json` 试试。",
            provider="siliconflow",
            model="",
        )

        self.assertEqual(prepared.display_text, "好的，那发一个 `config.json` 试试。")
        self.assertEqual(prepared.tts_text, "好的，那发一个 config.json 试试。")
        self.assertEqual(prepared.codes, [])

    def test_real_inline_code_is_still_appended_as_code_reference(self):
        prepared = self._sanitizer().prepare(
            "用 `x = 1` 试试",
            provider="siliconflow",
            model="",
        )

        self.assertEqual(prepared.display_text, "用 `x = 1` 试试")
        self.assertEqual(prepared.tts_text, "用 x = 1 试试")
        self.assertEqual(prepared.codes, ["`x = 1`"])


if __name__ == "__main__":
    unittest.main()
