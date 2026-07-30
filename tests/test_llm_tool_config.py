"""Regression tests for the tts_speak configuration path."""

import ast
import importlib
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

ConfigManager = importlib.import_module(
    "astrbot_plugin_tts_emotion_router.core.config"
).ConfigManager
PLUGIN_VERSION = importlib.import_module(
    "astrbot_plugin_tts_emotion_router.core.constants"
).PLUGIN_VERSION


class LlmToolConfigTest(unittest.TestCase):
    """Verify LLM tool defaults and explicit overrides."""

    def test_missing_config_defaults_to_enabled(self) -> None:
        """Keep tts_speak available for existing installations."""
        config = ConfigManager({})

        self.assertTrue(config.is_llm_tool_enabled())
        self.assertEqual(config.get_llm_tool_max_chars(), 200)
        self.assertEqual(
            config.get("feature_policies")["llm_tool"],
            {"enable": True, "max_chars": 200},
        )

    def test_feature_policy_values_are_respected(self) -> None:
        """Read explicit values from the schema-backed configuration path."""
        config = ConfigManager(
            {
                "feature_policies": {"llm_tool": {"enable": False, "max_chars": 321}},
                "output_strategies": {"llm_tool": {"enable": True, "max_chars": 999}},
            }
        )

        self.assertFalse(config.is_llm_tool_enabled())
        self.assertEqual(config.get_llm_tool_max_chars(), 321)

    def test_schema_and_runtime_versions_match(self) -> None:
        """Keep published metadata and runtime constants synchronized."""
        schema = json.loads(
            (PROJECT_ROOT / "_conf_schema.json").read_text(encoding="utf-8")
        )
        metadata = (PROJECT_ROOT / "metadata.yaml").read_text(encoding="utf-8")

        self.assertTrue(
            schema["feature_policies"]["items"]["llm_tool"]["items"]["enable"][
                "default"
            ]
        )
        self.assertIn(f"version: {PLUGIN_VERSION}", metadata)

    def test_tts_speak_stops_after_direct_voice_send(self) -> None:
        """Prevent the tool result from triggering a duplicate text response."""
        module = ast.parse((PROJECT_ROOT / "main.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "tts_speak"
        )

        self.assertIsInstance(function.body[-2], ast.Expr)
        yield_node = function.body[-2].value
        self.assertIsInstance(yield_node, ast.Yield)
        self.assertIsInstance(yield_node.value, ast.Constant)
        self.assertIsNone(yield_node.value.value)
        self.assertIsInstance(function.body[-1], ast.Return)


if __name__ == "__main__":
    unittest.main()
