# -*- coding: utf-8 -*-
"""TTS 情绪路由插件入口。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import logging
import time
import heapq
from typing import Any, Dict, List, Optional, Tuple

from .core.compat import initialize_compat

initialize_compat()

from .core.compat import (
    import_astr_message_event,
    import_filter,
    import_message_components,
    import_context_and_star,
    import_astrbot_config,
    import_llm_response,
    import_result_content_type,
)

AstrMessageEvent = import_astr_message_event()
filter = import_filter()
Record, Plain = import_message_components()
Context, Star, register = import_context_and_star()
AstrBotConfig = import_astrbot_config()
LLMResponse = import_llm_response()
ResultContentType = import_result_content_type()

from .core.constants import (
    PLUGIN_ID,
    PLUGIN_NAME,
    PLUGIN_DESC,
    PLUGIN_VERSION,
    TEMP_DIR,
    EMOTIONS,
    MIMO_STYLE_LABEL_TO_CATEGORY,
    EMOTION_KEYWORDS,
    AUDIO_CLEANUP_TTL_SECONDS,
    SESSION_CLEANUP_INTERVAL_SECONDS,
    SESSION_MAX_IDLE_SECONDS,
    SESSION_MAX_COUNT,
    INFLIGHT_SIG_TTL_SECONDS,
    INFLIGHT_SIG_MAX_COUNT,
    DEFAULT_TEST_TEXT,
    MINIMAX_EXPRESSIVE_MODELS,
    MINIMAX_EXPRESSIVE_TAGS,
    MIMO_PERFORMANCE_TAGS,
)
from .core.session import SessionState
from .core.config import ConfigManager
from .core.marker import EmotionMarkerProcessor
from .core.tts_processor import TTSProcessor, TTSConditionChecker, TTSResultBuilder
from .core.segmented_tts import SegmentedTTSProcessor
from .core.text_splitter import TextSplitter
from .emotion.classifier import HeuristicClassifier
from .tts.provider_siliconflow import SiliconFlowTTS
from .tts.provider_minimax import MiniMaxTTS
from .tts.provider_mimo import MiMoTTS, MIMO_EMOTION_TAGS_REVERSE, MIMO_VOICES
from .utils.audio import ensure_dir, cleanup_dir
from .utils.extract import CodeAndLinkExtractor
from .utils.text_sanitizer import PreparedSpeechText, SpeechTextSanitizer

logger = logging.getLogger(__name__)
VOICE_ONLY_SUPPRESSION_TTL_SECONDS = 120
RECENT_SPOKEN_ASSISTANT_CONTEXT_TTL_SECONDS = 300
OUTPUT_MARKER_MODE_EXTRA = "_tts_emotion_router_output_marker_mode"
OUTPUT_MARKER_MODE_PRESERVE = "preserve_for_tts"
OUTPUT_MARKER_MODE_STRIP = "strip_visible"


@register(PLUGIN_ID, PLUGIN_NAME, PLUGIN_DESC, PLUGIN_VERSION)
class TTSEmotionRouter(Star):
    def __init__(self, context: Context, config: Optional[dict] = None):
        super().__init__(context)

        self._session_state: Dict[str, SessionState] = {}
        self._inflight_sigs: Dict[str, float] = {}
        self._background_tasks: List[asyncio.Task] = []
        self._cleanup_task_started = False

        self._init_config(config)
        self._init_components()
        ensure_dir(TEMP_DIR)

    async def terminate(self):
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._background_tasks.clear()

        if hasattr(self, "tts_client"):
            try:
                await self.tts_client.close()
            except Exception:
                pass

        self._session_state.clear()
        self._inflight_sigs.clear()

    # ---------------- config/runtime init ----------------

    def _init_config(self, config: Optional[dict]) -> None:
        if isinstance(config, AstrBotConfig):
            self.config = ConfigManager(config)
        else:
            self.config = ConfigManager(config or {})

        self.voice_map, self.speed_map = self._resolve_route_maps()
        self.global_enable = self.config.get_global_enable()
        self.enabled_umos = self.config.get_enabled_umos()
        self.disabled_umos = self.config.get_disabled_umos()
        self.prob = self.config.get_prob()
        self.text_limit = self.config.get_text_limit()
        self.cooldown = self.config.get_cooldown()
        self.allow_mixed = self.config.get_allow_mixed()
        self.show_references = self.config.get_show_references()
        self.segmented_tts_enabled = self.config.is_segmented_tts_enabled()
        self.segmented_min_chars = self.config.get_segmented_tts_min_segment_chars()

    def _resolve_route_maps(self) -> tuple[Dict[str, str], Dict[str, float]]:
        route_enabled = self.config.is_emotion_route_enabled()
        voice_map: Dict[str, str] = self.config.get_voice_map() if route_enabled else {}
        speed_map: Dict[str, float] = self.config.get_speed_map() if route_enabled else {}
        voice_map = voice_map or {}
        speed_map = speed_map or {}

        default_voice = self.config.get_default_voice()
        if default_voice and not voice_map.get("neutral"):
            voice_map["neutral"] = default_voice

        api_cfg = self.config.get_api_config()
        if "neutral" not in speed_map:
            speed_map["neutral"] = float(api_cfg.get("speed", 1.0))
        return voice_map, speed_map

    def _create_tts_client(self):
        api_cfg = self.config.get_api_config()
        provider = api_cfg.get("provider", "siliconflow")

        if provider == "minimax":
            return MiniMaxTTS(
                api_url=api_cfg["url"],
                api_key=api_cfg["key"],
                model=api_cfg["model"],
                fmt=api_cfg["format"],
                speed=api_cfg["speed"],
                voice_id=api_cfg.get("voice_id", ""),
                vol=api_cfg.get("vol", 1.0),
                pitch=api_cfg.get("pitch", 0),
                default_emotion=api_cfg.get("emotion", "neutral"),
                sample_rate=api_cfg.get("sample_rate", 32000),
                bitrate=api_cfg.get("bitrate", 128000),
                channel=api_cfg.get("channel", 1),
                output_format=api_cfg.get("output_format", "hex"),
                language_boost=api_cfg.get("language_boost", ""),
                proxy=api_cfg.get("proxy", ""),
                voice_modify=api_cfg.get("voice_modify", {}),
                timber_weights=api_cfg.get("timber_weights", api_cfg.get("timbre_weights", [])),
                subtitle_enable=api_cfg.get("subtitle_enable", False),
                pronunciation_dict=api_cfg.get("pronunciation_dict", {}),
                aigc_watermark=api_cfg.get("aigc_watermark", False),
                max_retries=api_cfg.get("max_retries", 2),
                timeout=api_cfg.get("timeout", 30),
            )

        if provider == "mimo":
            return MiMoTTS(
                api_url=api_cfg["url"],
                api_key=api_cfg["key"],
                model=api_cfg["model"],
                voice=api_cfg.get("voice", "mimo_default"),
                fmt=api_cfg["format"],
                speed=api_cfg["speed"],
                max_retries=api_cfg.get("max_retries", 2),
                timeout=api_cfg.get("timeout", 30),
                style_prompt=api_cfg.get("style_prompt", ""),
                overall_tone=api_cfg.get("overall_tone", ""),
                timbre_positioning=api_cfg.get("timbre_positioning", ""),
                persona_accent=api_cfg.get("persona_accent", ""),
                dialect=api_cfg.get("dialect", ""),
                role_play=api_cfg.get("role_play", ""),
                seed_text=api_cfg.get("seed_text", ""),
                sing_voice=api_cfg.get("sing_voice", ""),
                director_enable=api_cfg.get("director_enable", False),
                director_role=api_cfg.get("director_role", ""),
                director_scene=api_cfg.get("director_scene", ""),
                director_instruction=api_cfg.get("director_instruction", ""),
                director_context=api_cfg.get("director_context", ""),
            )

        return SiliconFlowTTS(
            api_cfg["url"],
            api_cfg["key"],
            api_cfg["model"],
            api_cfg["format"],
            api_cfg["speed"],
            gain=api_cfg["gain"],
            sample_rate=api_cfg["sample_rate"],
            max_retries=api_cfg.get("max_retries", 2),
            timeout=api_cfg.get("timeout", 30),
        )

    def _get_tts_engine_signature(self) -> Tuple:
        api_cfg = self.config.get_api_config()
        keys = (
            "provider", "url", "key", "model", "format", "speed", "gain", "sample_rate",
            "voice_id", "voice", "sing_voice", "overall_tone", "timbre_positioning", "persona_accent", "dialect", "role_play", "director_enable", "director_role", "director_scene", "director_instruction", "director_context", "performance_tags_enable", "performance_tags", "vol", "pitch", "emotion", "bitrate", "channel", "subtitle_enable",
            "output_format", "language_boost", "proxy", "voice_modify", "timber_weights",
            "pronunciation_dict", "aigc_watermark", "max_retries", "timeout",
        )
        return tuple((k, str(api_cfg.get(k))) for k in keys)

    def _init_components(self) -> None:
        self.tts_client = self._create_tts_client()
        self.heuristic_cls = HeuristicClassifier(keywords=self.config.get_emotion_keywords())

        self.emo_marker_enable = self.config.is_marker_enabled()
        marker_tag = self.config.get_marker_tag()
        self.marker_processor = EmotionMarkerProcessor(tag=marker_tag, enabled=self.emo_marker_enable)

        self.extractor = CodeAndLinkExtractor()
        self.text_sanitizer = SpeechTextSanitizer(
            marker_processor=self.marker_processor,
            extractor=self.extractor,
        )
        self.tts_processor = TTSProcessor(
            tts_client=self.tts_client,
            voice_map=self.voice_map,
            speed_map=self.speed_map,
            heuristic_classifier=self.heuristic_cls,
        )
        self.condition_checker = TTSConditionChecker(
            prob=self.prob,
            text_limit=self.text_limit,
            text_min_limit=self.config.get_text_min_limit(),
            cooldown=self.cooldown,
            allow_mixed=self.allow_mixed,
        )
        self.result_builder = TTSResultBuilder(Plain, Record)

        self._init_segmented_tts()
        self._tts_engine_signature = self._get_tts_engine_signature()

        self.tts = self.tts_client
        self.emo_marker_tag = marker_tag
        self._emo_marker_re = self.marker_processor._marker_strict_re
        self._emo_marker_re_any = self.marker_processor._marker_any_re
        self._emo_head_token_re = self.marker_processor._head_token_re
        self._emo_head_anylabel_re = self.marker_processor._head_anylabel_re
        self._emo_kw = EMOTION_KEYWORDS

    def _init_segmented_tts(self) -> None:
        splitter = TextSplitter(
            split_pattern=self.config.get_segmented_tts_split_pattern(),
            smart_mode=True,
            max_segments=self.config.get_segmented_tts_max_segments(),
            min_segment_length=self.config.get_segmented_tts_min_segment_length(),
        )
        self.segmented_tts_processor = SegmentedTTSProcessor(
            tts_processor=self.tts_processor,
            splitter=splitter,
            interval_mode=self.config.get_segmented_tts_interval_mode(),
            fixed_interval=self.config.get_segmented_tts_fixed_interval(),
            adaptive_buffer=self.config.get_segmented_tts_adaptive_buffer(),
            max_segments=self.config.get_segmented_tts_max_segments(),
            min_segment_length=self.config.get_segmented_tts_min_segment_length(),
        )

    def _update_components_from_config(self) -> None:
        self.prob = self.config.get_prob()
        self.text_limit = self.config.get_text_limit()
        self.cooldown = self.config.get_cooldown()
        self.allow_mixed = self.config.get_allow_mixed()
        self.show_references = self.config.get_show_references()
        self.segmented_tts_enabled = self.config.is_segmented_tts_enabled()
        self.segmented_min_chars = self.config.get_segmented_tts_min_segment_chars()

        new_signature = self._get_tts_engine_signature()
        if new_signature != getattr(self, "_tts_engine_signature", None):
            old_client = self.tts_client
            self.tts_client = self._create_tts_client()
            self.tts_processor.tts = self.tts_client
            self.tts = self.tts_client
            self._tts_engine_signature = new_signature
            if old_client is not None and old_client is not self.tts_client:
                self._schedule_client_close(old_client)

        self.condition_checker.prob = self.prob
        self.condition_checker.text_limit = self.text_limit
        self.condition_checker.text_min_limit = self.config.get_text_min_limit()
        self.condition_checker.cooldown = self.cooldown
        self.condition_checker.allow_mixed = self.allow_mixed

        self.voice_map, self.speed_map = self._resolve_route_maps()
        self.tts_processor.voice_map = self.voice_map
        self.tts_processor.speed_map = self.speed_map

        self.global_enable = self.config.get_global_enable()
        self.enabled_umos = self.config.get_enabled_umos()
        self.disabled_umos = self.config.get_disabled_umos()

        self.emo_marker_enable = self.config.is_marker_enabled()
        self.marker_processor.update_config(self.config.get_marker_tag(), self.emo_marker_enable)
        self._init_segmented_tts()

    # ---------------- session helpers ----------------

    def _sess_id(self, event: AstrMessageEvent) -> str:
        gid = ""
        try:
            gid = event.get_group_id()
        except Exception:
            gid = ""

        if gid and gid not in ("", "None", "null", "0"):
            return f"group_{gid}"
        return f"user_{event.get_sender_id()}"

    def _get_umo(self, event: AstrMessageEvent) -> str:
        try:
            umo = str(getattr(event, "unified_msg_origin", "") or "").strip()
            if umo:
                return umo
        except Exception:
            pass
        return self._sess_id(event)

    def _get_session_state(self, sid: str) -> SessionState:
        return self._session_state.setdefault(sid, SessionState())

    @staticmethod
    def _normalize_conversation_id(conversation_id: Optional[str]) -> Optional[str]:
        cleaned = str(conversation_id or "").strip()
        return cleaned or None

    async def _remember_spoken_assistant_text(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        conversation_id: Optional[str] = None,
    ) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        sid = self._get_umo(event)
        conversation_id = self._normalize_conversation_id(conversation_id)
        if conversation_id is None:
            conversation_id = self._normalize_conversation_id(
                await self._get_current_conversation_id(event)
            )
        self._get_session_state(sid).set_spoken_assistant_text(
            cleaned,
            conversation_id=conversation_id,
        )

    async def _queue_pending_spoken_assistant_text(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        conversation_id: Optional[str] = None,
    ) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        sid = self._get_umo(event)
        conversation_id = self._normalize_conversation_id(conversation_id)
        if conversation_id is None:
            conversation_id = self._normalize_conversation_id(
                await self._get_current_conversation_id(event)
            )
        self._get_session_state(sid).queue_pending_spoken(
            cleaned,
            conversation_id,
        )

    async def _get_recent_spoken_assistant_text(
        self,
        event: AstrMessageEvent,
    ) -> Optional[str]:
        sid = self._get_umo(event)
        st = self._session_state.get(sid)
        if not st:
            return None

        text = (st.last_spoken_assistant_text or "").strip()
        if not text:
            return None

        now_ts = time.time()
        if now_ts - st.last_spoken_assistant_time > RECENT_SPOKEN_ASSISTANT_CONTEXT_TTL_SECONDS:
            return None
        current_conversation_id = self._normalize_conversation_id(
            await self._get_current_conversation_id(event)
        )
        cached_conversation_id = self._normalize_conversation_id(
            st.last_spoken_assistant_conversation_id
        )
        if current_conversation_id != cached_conversation_id:
            logger.info(
                "skip recent spoken assistant context sid=%s current_cid=%s cached_cid=%s reason=conversation_mismatch",
                sid,
                current_conversation_id,
                cached_conversation_id,
            )
            return None
        return text

    @staticmethod
    def _contexts_have_assistant_text(contexts: Any, text: str) -> bool:
        cleaned = (text or "").strip()
        if not cleaned:
            return False

        if not isinstance(contexts, list):
            return False

        for item in reversed(contexts[-8:]):
            if not isinstance(item, dict):
                continue
            if item.get("role") != "assistant":
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip() == cleaned:
                return True
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if str(part.get("type", "")).strip() != "text":
                        continue
                    part_text = str(part.get("text", "") or "").strip()
                    if part_text:
                        text_parts.append(part_text)
                if "".join(text_parts).strip() == cleaned:
                    return True
                if " ".join(text_parts).strip() == cleaned:
                    return True
                if "\n".join(text_parts).strip() == cleaned:
                    return True
        return False

    async def _inject_recent_spoken_assistant_context(
        self,
        event: AstrMessageEvent,
        request: Any,
    ) -> None:
        spoken_text = await self._get_recent_spoken_assistant_text(event)
        if not spoken_text:
            return

        contexts = getattr(request, "contexts", None)
        if contexts is None:
            contexts = []
        elif isinstance(contexts, str):
            try:
                contexts = json.loads(contexts)
            except Exception:
                contexts = []

        if not isinstance(contexts, list):
            return
        if self._contexts_have_assistant_text(contexts, spoken_text):
            return

        contexts.append({"role": "assistant", "content": spoken_text, "_no_save": True})
        request.contexts = contexts
        logger.info(
            "inject recent spoken assistant context sid=%s text=%s",
            self._get_umo(event),
            spoken_text[:80],
        )

    async def _get_current_conversation_id(self, event: AstrMessageEvent) -> Optional[str]:
        manager = getattr(self.context, "conversation_manager", None)
        if manager is None:
            return None
        return await manager.get_curr_conversation_id(self._get_umo(event))

    async def _ensure_conversation_id(self, event: AstrMessageEvent) -> Optional[str]:
        manager = getattr(self.context, "conversation_manager", None)
        if manager is None:
            return None

        sid = self._get_umo(event)
        conversation_id = await manager.get_curr_conversation_id(sid)
        if conversation_id:
            return conversation_id
        return await manager.new_conversation(sid)

    def _track_background_task(self, coro, name: str) -> None:
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.append(task)

        def _cleanup_done(done_task: asyncio.Task) -> None:
            try:
                self._background_tasks.remove(done_task)
            except ValueError:
                pass

        task.add_done_callback(_cleanup_done)

    def _schedule_client_close(self, client: Any) -> None:
        async def _close() -> None:
            try:
                await client.close()
            except Exception:
                logger.debug("close stale tts client failed", exc_info=True)

        self._track_background_task(_close(), "tts_close_stale_client")

    async def _start_background_tasks(self) -> None:
        if self._cleanup_task_started:
            return
        self._cleanup_task_started = True

        self._track_background_task(self._periodic_audio_cleanup(), "tts_audio_cleanup")
        self._track_background_task(self._periodic_session_cleanup(), "tts_session_cleanup")

    async def _periodic_audio_cleanup(self) -> None:
        try:
            while True:
                await cleanup_dir(TEMP_DIR, ttl_seconds=AUDIO_CLEANUP_TTL_SECONDS)
                await asyncio.sleep(AUDIO_CLEANUP_TTL_SECONDS // 2)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("audio cleanup error: %s", e)

    async def _periodic_session_cleanup(self) -> None:
        try:
            while True:
                await asyncio.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)
                await self._cleanup_stale_sessions()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("session cleanup error: %s", e)

    async def _cleanup_stale_sessions(self) -> None:
        now = time.time()
        for sid, state in self._session_state.items():
            if state.clear_next_llm_plain_text_suppression_if_expired(now):
                logger.info("voice-only suppression cleanup sid=%s reason=session_stale_scan", sid)

        stale_sessions = {
            sid
            for sid, state in self._session_state.items()
            if now - state.last_ts > SESSION_MAX_IDLE_SECONDS
        }

        if len(self._session_state) > SESSION_MAX_COUNT:
            excess_count = len(self._session_state) - SESSION_MAX_COUNT
            candidates = (
                (state.last_ts, sid)
                for sid, state in self._session_state.items()
                if sid not in stale_sessions
            )
            for _, sid in heapq.nsmallest(excess_count, candidates):
                stale_sessions.add(sid)

        for sid in stale_sessions:
            self._session_state.pop(sid, None)

        self._cleanup_stale_inflight(now)

    def _build_inflight_sig(self, umo: str, text: str) -> str:
        digest = hashlib.sha1(f"{umo}:{text[:200]}".encode("utf-8")).hexdigest()
        return f"{umo}:{digest[:24]}"

    def _cleanup_stale_inflight(self, now: Optional[float] = None) -> None:
        if not self._inflight_sigs:
            return

        now_ts = now if now is not None else time.time()
        expired = [
            sig
            for sig, ts in self._inflight_sigs.items()
            if (now_ts - ts) > INFLIGHT_SIG_TTL_SECONDS
        ]
        for sig in expired:
            self._inflight_sigs.pop(sig, None)

        if len(self._inflight_sigs) > INFLIGHT_SIG_MAX_COUNT:
            excess = len(self._inflight_sigs) - INFLIGHT_SIG_MAX_COUNT
            oldest = heapq.nsmallest(excess, self._inflight_sigs.items(), key=lambda x: x[1])
            for sig, _ in oldest:
                self._inflight_sigs.pop(sig, None)

    # ---------------- text helpers ----------------

    def _normalize_text(self, text: str) -> str:
        return self.marker_processor.normalize_text(text)

    def _strip_emo_head_many(self, text: str) -> tuple[str, Optional[str]]:
        return self.marker_processor.strip_head_many(text)

    def _strip_any_visible_markers(self, text: str) -> str:
        return self.marker_processor.strip_all_visible_markers(text)

    def get_output_marker_mode(self, event: AstrMessageEvent) -> str:
        if not self.emo_marker_enable:
            return OUTPUT_MARKER_MODE_STRIP

        umo = self._get_umo(event)
        if self.config.is_voice_output_enabled_for_umo(umo):
            return OUTPUT_MARKER_MODE_PRESERVE

        return OUTPUT_MARKER_MODE_STRIP

    def publish_output_marker_mode(self, event: AstrMessageEvent) -> str:
        mode = self.get_output_marker_mode(event)
        try:
            event.set_extra(OUTPUT_MARKER_MODE_EXTRA, mode)
        except Exception:
            logger.debug("publish output marker mode failed", exc_info=True)
        return mode

    def sanitize_visible_output_text(self, text: str) -> str:
        normalized = self._normalize_text(text or "")
        stripped, _ = self._strip_emo_head_many(normalized)
        stripped = self._strip_any_visible_markers(stripped)
        visible = self._prepare_visible_text(stripped)
        return (visible or stripped or "").strip()

    def _is_minimax_provider(self) -> bool:
        return self.config.get_tts_provider() == "minimax"

    def _current_tts_model(self) -> str:
        api_cfg = self.config.get_api_config()
        return str(api_cfg.get("model", "") or "").strip().lower()

    def _supports_minimax_expressive_tags(self, model: Optional[str] = None) -> bool:
        current_model = str(model or self._current_tts_model() or "").strip().lower()
        return current_model in MINIMAX_EXPRESSIVE_MODELS

    def _build_minimax_guidance_instruction(self) -> str:
        expressive_supported = self._supports_minimax_expressive_tags()
        lines = [
            "以下规则只在回复预计会被转成语音时生效。",
            "可用换行表示段落切换，换行要自然，不要为了凑格式乱分段。",
            "如需停顿，只能在两段可发音文本之间插入 <#x#>，x 为秒数，可保留两位小数，不要连续使用多个停顿标记。",
        ]

        if expressive_supported:
            tags = "、".join(f"({tag})" for tag in MINIMAX_EXPRESSIVE_TAGS)
            lines.append(
                f"仅在自然需要时可使用以下语气词标签：{tags}。不要堆砌，不要把标签当正文解释。"
            )
        else:
            lines.append("当前模型不要输出 MiniMax 语气词标签，例如 (laughs)。")

        lines.append("这些控制符只服务语音效果，输出后继续正常作答，不要解释这些控制符。")
        return "\n".join(lines)

    def _should_inject_minimax_prompt(self, event: AstrMessageEvent) -> bool:
        if not self._is_minimax_provider():
            return False
        umo = self._get_umo(event)
        return self.config.is_voice_output_enabled_for_umo(umo)

    def _append_references_to_text(
        self,
        base_text: str,
        *,
        links: Optional[List[str]] = None,
        codes: Optional[List[str]] = None,
    ) -> str:
        text = (base_text or "").strip()
        if not self.show_references:
            return text

        extra_parts: List[str] = []
        if links:
            extra_parts.append("参考链接\n" + "\n".join(f"{i + 1}. {link}" for i, link in enumerate(links)))
        if codes:
            extra_parts.append("代码片段\n" + "\n".join(codes))

        return "\n\n".join(part for part in [text, *extra_parts] if part)

    def _build_display_result_chain(self, original_chain: List, display_text: str) -> List:
        new_chain = []
        if (display_text or "").strip():
            new_chain.append(Plain(text=display_text.strip()))
        for comp in original_chain:
            if not isinstance(comp, Plain):
                new_chain.append(comp)
        return new_chain

    def _build_fallback_result_chain(
        self,
        original_chain: List,
        display_text: str,
        *,
        links: Optional[List[str]] = None,
        codes: Optional[List[str]] = None,
    ) -> List:
        fallback_text = self._append_references_to_text(
            display_text,
            links=links,
            codes=codes,
        )
        return self._build_display_result_chain(original_chain, fallback_text)

    def _prepare_text_for_tts(self, text: str) -> PreparedSpeechText:
        api_cfg = self.config.get_api_config()
        return self.text_sanitizer.prepare(
            text or "",
            provider=str(api_cfg.get("provider", "") or ""),
            model=str(api_cfg.get("model", "") or ""),
            mimo_performance_tags_enable=bool(api_cfg.get("performance_tags_enable", False)),
            mimo_performance_tags=api_cfg.get("performance_tags", []),
        )

    def _prepare_visible_text(self, text: str) -> str:
        prepared = self._prepare_text_for_tts(text)
        return (prepared.display_text or "").strip()

    async def _build_manual_tts_chain(
        self,
        event: AstrMessageEvent,
        text: str,
    ) -> Tuple[bool, List, str]:
        prepared = self._prepare_text_for_tts(text)
        tts_text = (prepared.tts_text or "").strip()
        send_text = (prepared.display_text or "").strip()
        history_text = send_text or tts_text
        if not tts_text:
            return False, [], "没有可用于语音合成的文本。"

        umo = self._get_umo(event)
        self._publish_temporary_voice_directives(event, text)
        st = self._get_session_state(umo)
        proc_res = await self.tts_processor.process(tts_text, st)
        if not proc_res.success or not proc_res.audio_path:
            return False, [], f"TTS 合成失败：{proc_res.error or '未知错误'}"

        norm_path = self.tts_processor.normalize_audio_path(proc_res.audio_path)
        text_voice_enabled = self.config.is_text_voice_output_enabled_for_umo(umo)

        chain = []
        if text_voice_enabled and send_text:
            chain.append(Plain(text=send_text))
        chain.append(Record(file=norm_path))
        return True, chain, history_text

    async def _send_manual_tts(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        suppress_next_llm_plain_text: bool = False,
    ) -> str:
        ok, chain, history_or_error = await self._build_manual_tts_chain(event, text)
        if not ok:
            return history_or_error

        try:
            await event.send(event.chain_result(chain))
            if suppress_next_llm_plain_text:
                sid = self._get_umo(event)
                st = self._get_session_state(sid)
                st.mark_next_llm_plain_text_suppressed(ttl_seconds=VOICE_ONLY_SUPPRESSION_TTL_SECONDS)
                logger.info("voice-only suppression set sid=%s ttl=%ss", sid, VOICE_ONLY_SUPPRESSION_TTL_SECONDS)
            return "语音已发送。"
        except Exception as e:
            logging.error("manual tts send failed: %s", e)
            return f"发送失败：{e}"

    def _extract_temporary_performance_tags(self, text: str) -> List[str]:
        api_cfg = self.config.get_api_config()
        if self.config.get_tts_provider() != "mimo" or not bool(api_cfg.get("performance_tags_enable", False)):
            return []
        whitelist = {
            str(tag or "").strip()
            for tag in (api_cfg.get("performance_tags") or MIMO_PERFORMANCE_TAGS)
            if str(tag or "").strip()
        }
        if not whitelist:
            return []

        raw = str(text or "")
        matched: List[str] = []
        seen = set()

        # Explicit bracket tags in a director prompt: “开头带（叹气）”.
        for match in re.finditer(r"[（(]\s*([^（）()\n]{1,24})\s*[）)]", raw):
            tag = (match.group(1) or "").strip()
            if tag not in whitelist or tag in seen:
                continue
            seen.add(tag)
            matched.append(tag)

        # Natural-language director prompt: “叹气地说 / 带一点哽咽 / 轻笑一下”.
        # Prefer longer tags first so “声音颤抖” wins over “颤抖”.
        for tag in sorted(whitelist, key=len, reverse=True):
            if tag in seen:
                continue
            if re.search(re.escape(tag), raw):
                seen.add(tag)
                matched.append(tag)

        return matched

    def _extract_temporary_voice_directives(self, text: str) -> Tuple[Optional[str], Dict[str, str], Optional[str], Optional[str]]:
        """Extract one-turn MiMo style/emotion/voice directives from user text.

        Voice names are only accepted with an explicit speech context such as
        “用冰糖的音色说” or “用 Mia 的声音说”, avoiding accidental triggers in
        ordinary chat.
        """
        raw = str(text or "")
        if not raw.strip():
            return None, {}, None, None

        pending_emotion: Optional[str] = None
        for zh_label, emotion_key in MIMO_EMOTION_TAGS_REVERSE.items():
            if emotion_key in EMOTIONS and zh_label in raw:
                pending_emotion = emotion_key

        style_overrides: Dict[str, str] = {}
        for label, category in MIMO_STYLE_LABEL_TO_CATEGORY.items():
            if label in raw:
                style_overrides[category] = label

        pending_voice: Optional[str] = None
        for voice_name in sorted(MIMO_VOICES.keys(), key=len, reverse=True):
            if voice_name == "mimo_default":
                continue
            # Strong context only: “用/换成/切到 + voice + 的? + 音色/声音/声线/嗓音”.
            pattern = rf"(?:用|换成|切到|改成|使用|采用)\s*{re.escape(voice_name)}\s*(?:的)?\s*(?:音色|声音|声线|嗓音)"
            if re.search(pattern, raw, re.I):
                pending_voice = voice_name
                break

        pending_director: Optional[str] = None
        director_patterns = (
            r"(?:用这种语音风格|语音导演|语音指导|本次语音指导|临时语音指导|按这个风格说|按这种风格说)\s*[:：]\s*(?P<prompt>.+)",
            r'(?:用|按)\s*(?P<prompt>[^。！？!?"“”]{2,80}?)\s*(?:的)?\s*(?:语音风格|说话风格|表演方式)\s*(?:说|讲|朗读)',
        )
        for pattern in director_patterns:
            m = re.search(pattern, raw, re.I | re.S)
            if m:
                candidate = (m.group("prompt") or "").strip()
                candidate = re.split(r"(?:\n|。|！|？|!|\?)", candidate, maxsplit=1)[0].strip()
                if candidate:
                    pending_director = candidate[:300]
                    break

        return pending_emotion, style_overrides, pending_voice, pending_director

    def _publish_temporary_voice_directives(self, event: AstrMessageEvent, text: str) -> None:
        # MiMo-only: these directives use MiMo preset voices and Chinese style tags.
        # Do not leak them to MiniMax/SiliconFlow where they may be invalid voice ids.
        if self.config.get_tts_provider() != "mimo":
            return

        emotion, style_overrides, pending_voice, pending_director = self._extract_temporary_voice_directives(text)
        performance_tags = self._extract_temporary_performance_tags(pending_director or "")
        if not emotion and not style_overrides and not pending_voice and not pending_director and not performance_tags:
            return
        st = self._get_session_state(self._get_umo(event))
        if emotion:
            st.pending_emotion = emotion
        if style_overrides:
            st.set_pending_style_overrides(style_overrides)
        if pending_voice:
            st.set_pending_voice(pending_voice)
        if pending_director:
            st.set_pending_director_prompt(pending_director)
        if performance_tags:
            st.set_pending_performance_tags(performance_tags)
        logger.info(
            "temporary MiMo voice directives sid=%s emotion=%s styles=%s voice=%s director=%s performance_tags=%s",
            self._get_umo(event),
            emotion,
            style_overrides,
            pending_voice,
            bool(pending_director),
            performance_tags,
        )

    # ---------------- llm hooks ----------------

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, request):
        try:
            user_prompt = str(getattr(request, "prompt", "") or "")
            if user_prompt:
                self._publish_temporary_voice_directives(event, user_prompt)
            await self._inject_recent_spoken_assistant_context(event, request)
            marker_mode = self.publish_output_marker_mode(event)
            if not self._should_inject_minimax_prompt(event):
                return

            sp = getattr(request, "system_prompt", "") or ""
            pp = getattr(request, "prompt", "") or ""
            injected_parts: List[str] = []

            if (
                marker_mode == OUTPUT_MARKER_MODE_PRESERVE
                and self.emo_marker_enable
                and not self.marker_processor.is_marker_present(sp, pp)
            ):
                prompt_hint = self.config.get_marker_prompt_hint()
                if prompt_hint:
                    injected_parts.append(prompt_hint)
                injected_parts.append(self.marker_processor.build_injection_instruction())

            injected_parts.append(self._build_minimax_guidance_instruction())
            request.system_prompt = "\n".join(part for part in [*injected_parts, sp] if part).strip()
        except Exception as e:
            logger.error("on_llm_request failed: %s", e)

    @filter.on_llm_response(priority=1)
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        if not self.emo_marker_enable:
            return

        label: Optional[str] = None
        cached_text: Optional[str] = None
        rc = getattr(response, "result_chain", None)
        chain = getattr(rc, "chain", None)

        if rc and hasattr(rc, "chain") and chain:
            try:
                new_chain = []
                cleaned_once = False
                for comp in chain:
                    if (not cleaned_once and isinstance(comp, Plain) and getattr(comp, "text", None)):
                        t0 = self._normalize_text(comp.text)
                        t, l2 = self._strip_emo_head_many(t0)
                        if l2 in EMOTIONS and label is None:
                            label = l2
                        if t:
                            comp.text = t
                            new_chain.append(comp)
                            cached_text = t or cached_text
                        cleaned_once = True
                    else:
                        new_chain.append(comp)
                rc.chain = new_chain
                text = getattr(response, "completion_text", None)
                if isinstance(text, str) and text.strip():
                    cached_text = text
            except Exception as e:
                logging.warning("strip result_chain failed: %s", e)
        else:
            try:
                text = getattr(response, "completion_text", None)
                if isinstance(text, str) and text.strip():
                    t0 = self._normalize_text(text)
                    cleaned, l1 = self._strip_emo_head_many(t0)
                    if l1 in EMOTIONS:
                        label = l1
                    response.completion_text = cleaned
                    try:
                        setattr(response, "_completion_text", cleaned)
                    except Exception:
                        pass
                    cached_text = cleaned or cached_text
            except Exception as e:
                logging.warning("strip completion_text failed: %s", e)

        visible_text = (cached_text or "").strip()
        if visible_text:
            try:
                prepared_visible_text = self._prepare_visible_text(visible_text)
                if prepared_visible_text:
                    visible_text = prepared_visible_text
            except Exception as e:
                logging.warning("prepare visible history text failed: %s", e)

        try:
            umo = self._get_umo(event)
            st = self._get_session_state(umo)
            if label in EMOTIONS and not st.pending_emotion:
                st.pending_emotion = label
            if visible_text:
                st.set_assistant_text(visible_text)
        except Exception as e:
            logging.error("update session state failed: %s", e)

    @filter.on_decorating_result(priority=999)
    async def _final_strip_markers(self, event: AstrMessageEvent):
        if not self.emo_marker_enable:
            return

        try:
            result = event.get_result()
            if not result or not hasattr(result, "chain"):
                return
            for comp in list(result.chain):
                if isinstance(comp, Plain) and getattr(comp, "text", None):
                    comp.text = self._strip_any_visible_markers(comp.text)
        except Exception as e:
            logging.error("final marker cleanup failed: %s", e)

    @filter.on_decorating_result(priority=-1000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        await self._start_background_tasks()
        self._cleanup_stale_inflight()

        try:
            if hasattr(event, "is_stopped") and event.is_stopped():
                return
        except Exception:
            pass

        try:
            result = event.get_result()
            if not result:
                return

            try:
                is_llm_response = result.is_llm_result()
            except Exception:
                is_llm_response = getattr(result, "result_content_type", None) == ResultContentType.LLM_RESULT

            if not is_llm_response:
                return
            if not hasattr(result, "chain") or result.chain is None:
                result.chain = []
        except Exception as e:
            logging.warning("inspect response failed: %s", e)
            return

        umo = self._get_umo(event)
        st = self._session_state.get(umo)
        if st:
            now_ts = time.time()
            if st.clear_next_llm_plain_text_suppression_if_expired(now_ts):
                logger.info("voice-only suppression cleanup sid=%s reason=decorating_expired", umo)
            elif st.consume_next_llm_plain_text_suppression(now_ts):
                plain_removed = sum(1 for comp in result.chain if isinstance(comp, Plain))
                result.chain = [comp for comp in result.chain if not isinstance(comp, Plain)]
                logger.info("voice-only suppression consumed sid=%s plain_removed=%d", umo, plain_removed)
                return

        if not result.chain:
            return

        try:
            new_chain = []
            for comp in result.chain:
                if isinstance(comp, Plain) and getattr(comp, "text", None):
                    t0 = self._normalize_text(comp.text)
                    t, _ = self._strip_emo_head_many(t0)
                    t = self._strip_any_visible_markers(t)
                    if t:
                        new_chain.append(Plain(text=t))
                else:
                    new_chain.append(comp)
            result.chain = new_chain
        except Exception as e:
            logging.warning("marker strip failed: %s", e)

        text_parts = [c.text.strip() for c in result.chain if isinstance(c, Plain) and c.text.strip()]
        if not text_parts:
            return

        text = self._normalize_text(" ".join(text_parts))
        prepared = self._prepare_text_for_tts(text)
        tts_text = (prepared.tts_text or "").strip()
        display_text = (prepared.display_text or "").strip()
        send_text = self._append_references_to_text(
            display_text,
            links=prepared.links,
            codes=prepared.codes,
        )
        display_chain = self._build_fallback_result_chain(
            result.chain,
            display_text,
            links=prepared.links,
            codes=prepared.codes,
        )

        if not self.config.is_voice_output_enabled_for_umo(umo):
            result.chain = display_chain
            return

        if not tts_text:
            result.chain = display_chain
            return

        st = self._get_session_state(umo)
        allowed_components = {"Plain", "At", "Reply", "Image", "Face"}
        has_non_plain = any(type(c).__name__ not in allowed_components for c in result.chain)

        check_res = self.condition_checker.check_all(
            tts_text,
            st,
            has_non_plain,
            enable_probability=self.config.is_probability_output_enabled_for_umo(umo),
        )
        if not check_res.passed:
            result.chain = display_chain
            return

        sig = self._build_inflight_sig(umo, tts_text)
        if sig in self._inflight_sigs:
            result.chain = display_chain
            return
        self._inflight_sigs[sig] = time.time()

        try:
            text_voice_enabled = st.text_voice_enabled if st.text_voice_enabled is not None else self.config.is_text_voice_output_enabled_for_umo(umo)

            segmented_enabled = (
                self.segmented_tts_enabled
                and self.config.is_segmented_output_enabled_for_umo(umo)
                and self.segmented_tts_processor.should_use_segmented(tts_text, self.segmented_min_chars)
            )

            if segmented_enabled:
                seg_res = await self.segmented_tts_processor.process_only(tts_text, st)
                if seg_res.successful_segments:
                    out_chain = []
                    if text_voice_enabled and send_text:
                        out_chain.append(Plain(text=send_text))
                    for seg in seg_res.successful_segments:
                        if not seg.audio_path:
                            continue
                        out_chain.append(Record(file=self.tts_processor.normalize_audio_path(seg.audio_path)))
                    for comp in result.chain:
                        if not isinstance(comp, Plain):
                            out_chain.append(comp)
                    if out_chain:
                        result.chain = out_chain
                        if send_text:
                            await self._queue_pending_spoken_assistant_text(
                                event,
                                send_text,
                            )
                        return

            proc_res = await self.tts_processor.process(tts_text, st)
            if proc_res.success and proc_res.audio_path:
                result.chain = self.result_builder.build(
                    original_chain=result.chain,
                    audio_path=self.tts_processor.normalize_audio_path(proc_res.audio_path),
                    send_text=send_text,
                    text_voice_enabled=text_voice_enabled,
                )
                if send_text:
                    await self._queue_pending_spoken_assistant_text(
                        event,
                        send_text,
                    )
            else:
                result.chain = display_chain
        finally:
            self._inflight_sigs.pop(sig, None)

    # ---------------- history ----------------

    async def _ensure_history_saved(self, event: AstrMessageEvent) -> None:
        try:
            umo = self._get_umo(event)
            st = self._session_state.get(umo)
            if not st:
                return

            text, conversation_id = st.consume_pending_history()
            if not text:
                return

            ok = await self._append_assistant_text_to_history(
                event,
                text,
                conversation_id=conversation_id,
                create_if_missing=not bool(conversation_id),
            )
            if not ok:
                st.queue_pending_history(text, conversation_id)
        except Exception as e:
            logging.debug("ensure_history_saved error: %s", e)

    async def _append_assistant_text_to_history(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        conversation_id: Optional[str] = None,
        create_if_missing: bool = False,
    ) -> bool:
        try:
            cleaned = (text or "").strip()
            if not cleaned:
                return False

            manager = getattr(self.context, "conversation_manager", None)
            if manager is None:
                return False

            sid = self._get_umo(event)
            target_cid = (conversation_id or "").strip() or await manager.get_curr_conversation_id(sid)
            if not target_cid:
                if not create_if_missing:
                    logger.info("skip assistant history write sid=%s reason=no_conversation", sid)
                    return False
                target_cid = await manager.new_conversation(sid)

            conversation = await manager.get_conversation(sid, target_cid)
            if conversation is None and create_if_missing:
                target_cid = await manager.new_conversation(sid)
                conversation = await manager.get_conversation(sid, target_cid)
            if conversation is None:
                logger.warning(
                    "assistant history write failed sid=%s cid=%s reason=conversation_missing",
                    sid,
                    target_cid,
                )
                return False

            raw_history = getattr(conversation, "history", "[]") or "[]"
            try:
                history = json.loads(raw_history)
                if not isinstance(history, list):
                    raise TypeError("conversation history is not a list")
            except Exception:
                logger.warning(
                    "conversation history parse failed sid=%s cid=%s; reset to empty history",
                    sid,
                    target_cid,
                )
                history = []

            history.append({"role": "assistant", "content": cleaned})
            await manager.update_conversation(
                sid,
                target_cid,
                history=history,
            )
            return True
        except Exception:
            return False

    # ---------------- after message sent ----------------

    if hasattr(filter, "after_message_sent"):

        @filter.after_message_sent(priority=-1000)
        async def after_message_sent(self, event: AstrMessageEvent):
            try:
                result = event.get_result()
                if not result:
                    return

                chain = getattr(result, "chain", None) or []
                if any(isinstance(c, Record) for c in chain):
                    umo = self._get_umo(event)
                    st = self._session_state.get(umo)
                    pending_history_text = (st.pending_history_text or "").strip() if st else ""
                    pending_history_conversation_id = (
                        st.pending_history_conversation_id if st else None
                    )
                    if pending_history_text:
                        await self._remember_spoken_assistant_text(
                            event,
                            pending_history_text,
                            conversation_id=pending_history_conversation_id,
                        )
                        if st:
                            st.clear_pending_spoken()
                    elif st:
                        pending_spoken_text, pending_spoken_conversation_id = (
                            st.consume_pending_spoken()
                        )
                        if pending_spoken_text:
                            await self._remember_spoken_assistant_text(
                                event,
                                pending_spoken_text,
                                conversation_id=pending_spoken_conversation_id,
                            )
                    await self._ensure_history_saved(event)

                try:
                    is_llm_response = result.is_llm_result()
                except Exception:
                    is_llm_response = getattr(result, "result_content_type", None) == ResultContentType.LLM_RESULT

                umo = self._get_umo(event)
                st = self._session_state.get(umo)
                if st and st.clear_next_llm_plain_text_suppression_if_expired():
                    logger.info("voice-only suppression cleanup sid=%s reason=after_message_sent_expired", umo)
                if st and is_llm_response and st.clear_next_llm_plain_text_suppression():
                    logger.info("voice-only suppression cleanup sid=%s reason=after_message_sent", umo)
            except Exception as e:
                logging.error("after_message_sent error: %s", e)

    else:

        async def after_message_sent(self, event: AstrMessageEvent):
            _ = event
            return

    # ---------------- commands & llm tool ----------------

    async def _switch_voice_output_for_current_umo(
        self,
        event: AstrMessageEvent,
        *,
        enable: bool,
    ) -> str:
        umo = self._get_umo(event)
        policy = self.config.get_feature_policy("voice_output")
        if enable and not bool(policy.get("enable", True)):
            await self.config.set_voice_output_enable_async(True)
            policy = self.config.get_feature_policy("voice_output")
        mode = policy.get("mode", "blacklist")

        if mode == "whitelist":
            if enable:
                await self.config.add_to_enabled_umos_async(umo)
            else:
                await self.config.remove_from_enabled_umos_async(umo)
        else:
            if enable:
                await self.config.remove_from_disabled_umos_async(umo)
            else:
                await self.config.add_to_disabled_umos_async(umo)

        self._update_components_from_config()
        return umo

    @filter.command("tts_on", priority=1)
    async def tts_on(self, event: AstrMessageEvent):
        umo = await self._switch_voice_output_for_current_umo(event, enable=True)
        yield event.plain_result(f"当前会话已开启语音输出。UMO={umo}")

    @filter.command("tts_off", priority=1)
    async def tts_off(self, event: AstrMessageEvent):
        umo = await self._switch_voice_output_for_current_umo(event, enable=False)
        yield event.plain_result(f"当前会话已关闭语音输出。UMO={umo}")

    @filter.command("tts_all_on", priority=1)
    async def tts_all_on(self, event: AstrMessageEvent):
        _ = event
        await self.config.set_voice_output_enable_async(True)
        self._update_components_from_config()
        yield event.plain_result("已开启全局自动语音输出。")

    @filter.command("tts_all_off", priority=1)
    async def tts_all_off(self, event: AstrMessageEvent):
        _ = event
        await self.config.set_voice_output_enable_async(False)
        self._update_components_from_config()
        yield event.plain_result("已关闭全局自动语音输出。可用 tts_say 或 tts_speak 按需发语音。")

    @filter.command("tts_status", priority=1)
    async def tts_status(self, event: AstrMessageEvent):
        umo = self._get_umo(event)
        provider = self.config.get_tts_provider()
        voice_enabled = self.config.is_voice_output_enabled_for_umo(umo)
        text_voice_enabled = self.config.is_text_voice_output_enabled_for_umo(umo)
        segmented_enabled = self.config.is_segmented_output_enabled_for_umo(umo)
        probability_enabled = self.config.is_probability_output_enabled_for_umo(umo)

        msg = (
            f"语音服务商: {provider}\n"
            f"UMO: {umo}\n"
            f"自动语音输出: {voice_enabled}\n"
            f"文字+语音同发: {text_voice_enabled}\n"
            f"分段语音输出: {segmented_enabled}\n"
            f"概率语音输出: {probability_enabled}\n"
            f"prob: {self.prob}, text_limit: {self.text_limit}, cooldown: {self.cooldown}s\n"
            "提示: 在聊天中发送 /sid 可获取当前 UMO。"
        )
        yield event.plain_result(msg)

    @filter.command("tts_say", priority=1)
    async def tts_say(self, event: AstrMessageEvent, *, text: Optional[str] = None):
        content = (text or DEFAULT_TEST_TEXT).strip()
        if not content:
            content = DEFAULT_TEST_TEXT

        ok, chain, history_or_error = await self._build_manual_tts_chain(event, content)
        if not ok:
            yield event.plain_result(history_or_error)
            return

        history_text = history_or_error.strip()
        conversation_id = await self._ensure_conversation_id(event)
        if history_text:
            sid = self._get_umo(event)
            self._get_session_state(sid).queue_pending_history(
                history_text,
                conversation_id,
            )

        yield event.chain_result(chain)

        if history_text and not hasattr(filter, "after_message_sent"):
            await self._remember_spoken_assistant_text(
                event,
                history_text,
                conversation_id=conversation_id,
            )
            await self._ensure_history_saved(event)

    if hasattr(filter, "llm_tool"):

        @filter.llm_tool(name="tts_speak")
        async def tts_speak(self, event: AstrMessageEvent, text: str):
            """按需输出语音（手动触发，不受自动语音总开关影响）。

            Args:
                text(string): 需要合成并发送的文本内容。

            Returns:
                string: 发送结果文本（成功/失败说明）。
            """
            content = (text or "").strip()
            if not content:
                yield "文本为空"
                return

            ok, chain, history_or_error = await self._build_manual_tts_chain(event, content)
            if not ok:
                logging.warning("tts_speak synthesis failed: %s", history_or_error)
                event.clear_result()
                # Do not expose the low-level failure text to the LLM.  If the
                # model sees "audio generation failed", it tends to roleplay a
                # broken voice ("嗓子被封印") and sends confusing fallback text.
                yield None
                yield "语音请求已处理。"
                return

            history_text = history_or_error.strip()
            sid = self._get_umo(event)
            st = self._get_session_state(sid)
            conversation_id = await self._get_current_conversation_id(event)
            if history_text:
                st.queue_pending_history(history_text, conversation_id)
            try:
                await event.send(event.chain_result(chain))
            except Exception as e:
                st.clear_pending_history()
                logging.error("tts_speak send failed: %s", e)
                yield f"发送失败：{e}"
                return

            if history_text:
                await self._remember_spoken_assistant_text(
                    event,
                    history_text,
                    conversation_id=conversation_id,
                )
                if not hasattr(filter, "after_message_sent"):
                    await self._ensure_history_saved(event)
            event.clear_result()
            yield None
            yield history_text or "语音已发送。"
            return
