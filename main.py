# -*- coding: utf-8 -*-
"""TTS 情绪路由插件入口。"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
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
    EMOTION_KEYWORDS,
    AUDIO_CLEANUP_TTL_SECONDS,
    SESSION_CLEANUP_INTERVAL_SECONDS,
    SESSION_MAX_IDLE_SECONDS,
    SESSION_MAX_COUNT,
    INFLIGHT_SIG_TTL_SECONDS,
    INFLIGHT_SIG_MAX_COUNT,
    DEFAULT_TEST_TEXT,
    HISTORY_WRITE_DELAY,
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
from .utils.audio import ensure_dir, cleanup_dir
from .utils.extract import CodeAndLinkExtractor, ProcessedText

logger = logging.getLogger(__name__)
VOICE_ONLY_SUPPRESSION_TTL_SECONDS = 120


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
                subtitle_enable=api_cfg.get("subtitle_enable", False),
                pronunciation_dict=api_cfg.get("pronunciation_dict", {}),
                max_retries=api_cfg.get("max_retries", 2),
                timeout=api_cfg.get("timeout", 30),
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
            "voice_id", "vol", "pitch", "emotion", "bitrate", "channel", "subtitle_enable",
            "pronunciation_dict", "max_retries", "timeout",
        )
        return tuple((k, str(api_cfg.get(k))) for k in keys)

    def _init_components(self) -> None:
        self.tts_client = self._create_tts_client()
        self.heuristic_cls = HeuristicClassifier(keywords=self.config.get_emotion_keywords())

        self.emo_marker_enable = self.config.is_marker_enabled()
        marker_tag = self.config.get_marker_tag()
        self.marker_processor = EmotionMarkerProcessor(tag=marker_tag, enabled=self.emo_marker_enable)

        self.extractor = CodeAndLinkExtractor()
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

    def _prepare_text_for_tts(self, text: str) -> Tuple[str, str]:
        text = self._normalize_text(text or "")
        text, _ = self._strip_emo_head_many(text)
        processed: ProcessedText = self.extractor.process_text(text)
        tts_text = (processed.speak_text or "").strip()
        send_text = (processed.clean_text or "").strip()
        return tts_text, send_text

    async def _build_manual_tts_chain(
        self,
        event: AstrMessageEvent,
        text: str,
    ) -> Tuple[bool, List, str]:
        tts_text, send_text = self._prepare_text_for_tts(text)
        if not tts_text:
            return False, [], "没有可用于语音合成的文本。"

        umo = self._get_umo(event)
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
        if send_text:
            st.set_assistant_text(send_text)
        return True, chain, "ok"

    async def _send_manual_tts(
        self,
        event: AstrMessageEvent,
        text: str,
        *,
        suppress_next_llm_plain_text: bool = False,
    ) -> str:
        ok, chain, msg = await self._build_manual_tts_chain(event, text)
        if not ok:
            return msg

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

    # ---------------- llm hooks ----------------

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, request):
        _ = event
        if not self.emo_marker_enable:
            return

        try:
            sp = getattr(request, "system_prompt", "") or ""
            pp = getattr(request, "prompt", "") or ""
            if not self.marker_processor.is_marker_present(sp, pp):
                request.system_prompt = (self.marker_processor.build_injection_instruction() + "\n" + sp).strip()
        except Exception as e:
            logger.error("on_llm_request failed: %s", e)

    @filter.on_llm_response(priority=1)
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        if not self.emo_marker_enable:
            return

        label: Optional[str] = None
        cached_text: Optional[str] = None

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

        try:
            rc = getattr(response, "result_chain", None)
            if rc and hasattr(rc, "chain") and rc.chain:
                new_chain = []
                cleaned_once = False
                for comp in rc.chain:
                    if (not cleaned_once and isinstance(comp, Plain) and getattr(comp, "text", None)):
                        t0 = self._normalize_text(comp.text)
                        t, l2 = self._strip_emo_head_many(t0)
                        if l2 in EMOTIONS and label is None:
                            label = l2
                        if t:
                            new_chain.append(Plain(text=t))
                            cached_text = t or cached_text
                        cleaned_once = True
                    else:
                        new_chain.append(comp)
                rc.chain = new_chain
        except Exception as e:
            logging.warning("strip result_chain failed: %s", e)

        try:
            umo = self._get_umo(event)
            st = self._get_session_state(umo)
            if label in EMOTIONS:
                st.pending_emotion = label
            if cached_text and cached_text.strip():
                st.set_assistant_text(cached_text.strip())
        except Exception as e:
            logging.error("update session state failed: %s", e)

        try:
            if cached_text and cached_text.strip():
                ok = await self._append_assistant_text_to_history(event, cached_text.strip())
                if not ok:
                    asyncio.create_task(self._delayed_history_write(event, cached_text.strip(), delay=HISTORY_WRITE_DELAY))
        except Exception as e:
            logging.error("append history failed: %s", e)

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

        if not self.config.is_voice_output_enabled_for_umo(umo):
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
        text, _ = self._strip_emo_head_many(text)

        processed: ProcessedText = self.extractor.process_text(text)
        tts_text = (processed.speak_text or "").strip()
        clean_text = (processed.clean_text or "").strip()
        links = processed.links
        codes = processed.codes

        send_text = clean_text
        if self.show_references:
            if links:
                send_text += "\n\n参考链接\n" + "\n".join(f"{i + 1}. {link}" for i, link in enumerate(links))
            if codes:
                send_text += "\n\n代码片段\n" + "\n".join(codes)

        if not tts_text:
            result.chain = [Plain(text=send_text)]
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
            if "mixed content" in check_res.reason:
                result.chain = [Plain(text=send_text)] + [c for c in result.chain if not isinstance(c, Plain)]
            return

        sig = self._build_inflight_sig(umo, tts_text)
        if sig in self._inflight_sigs:
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
                            st.set_assistant_text(send_text)
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
                    st.set_assistant_text(send_text)
            else:
                result.chain = [Plain(text=send_text)]
        finally:
            self._inflight_sigs.pop(sig, None)

    # ---------------- history ----------------

    async def _ensure_history_saved(self, event: AstrMessageEvent) -> None:
        try:
            umo = self._get_umo(event)
            st = self._session_state.get(umo)
            if not st or not st.assistant_text:
                return

            text = st.assistant_text
            st.assistant_text = None
            await self._append_assistant_text_to_history(event, text)
        except Exception as e:
            logging.debug("ensure_history_saved error: %s", e)

    async def _invoke_maybe_async(self, method: Any, *args: Any) -> Any:
        result = method(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _write_history_via_conversation_manager(self, sid: str, text: str) -> bool:
        manager = getattr(self.context, "conversation_manager", None)
        if manager is None:
            return False

        # Prefer manager-level assistant append methods; fallback to generic message APIs.
        method_candidates = [
            ("append_assistant_response", (sid, text)),
            ("append_assistant_message", (sid, text)),
            ("add_assistant_message", (sid, text)),
            ("append_message", (sid, "assistant", text)),
            ("add_message", (sid, "assistant", text)),
        ]
        for method_name, args in method_candidates:
            method = getattr(manager, method_name, None)
            if not callable(method):
                continue
            try:
                result = await self._invoke_maybe_async(method, *args)
                if result is False:
                    continue
                return True
            except TypeError:
                continue
            except Exception:
                logger.debug("conversation_manager.%s failed", method_name, exc_info=True)

        # Some AstrBot versions expose per-conversation objects.
        for getter_name in ("get_conversation", "get_or_create_conversation", "get_session"):
            getter = getattr(manager, getter_name, None)
            if not callable(getter):
                continue
            try:
                conv = await self._invoke_maybe_async(getter, sid)
            except Exception:
                continue
            if conv is None:
                continue

            conv_candidates = [
                ("append_assistant_response", (text,)),
                ("append_assistant_message", (text,)),
                ("add_assistant_message", (text,)),
                ("append_message", ("assistant", text)),
                ("add_message", ("assistant", text)),
            ]
            for method_name, args in conv_candidates:
                method = getattr(conv, method_name, None)
                if not callable(method):
                    continue
                try:
                    result = await self._invoke_maybe_async(method, *args)
                    if result is False:
                        continue
                    return True
                except TypeError:
                    continue
                except Exception:
                    logger.debug("conversation object %s failed", method_name, exc_info=True)

        return False

    async def _write_history_via_provider(self, sid: str, text: str) -> bool:
        provider = getattr(self.context, "llm_provider", None)
        try:
            if provider is None and hasattr(self.context, "get_llm_provider"):
                provider = self.context.get_llm_provider()
        except Exception:
            provider = None

        if provider is None:
            return False

        method_candidates = [
            ("append_assistant_response", (sid, text)),
            ("append_message", (sid, "assistant", text)),
            ("add_message", (sid, "assistant", text)),
        ]
        for method_name, args in method_candidates:
            method = getattr(provider, method_name, None)
            if not callable(method):
                continue
            try:
                result = await self._invoke_maybe_async(method, *args)
                if result is False:
                    continue
                return True
            except TypeError:
                continue
            except Exception:
                logger.debug("provider.%s failed", method_name, exc_info=True)
        return False

    async def _append_assistant_text_to_history(self, event: AstrMessageEvent, text: str) -> bool:
        try:
            if not text or not text.strip():
                return False
            sid = self._get_umo(event)
            if await self._write_history_via_conversation_manager(sid, text):
                return True
            return await self._write_history_via_provider(sid, text)
        except Exception:
            return False

    async def _delayed_history_write(
        self,
        event: AstrMessageEvent,
        text: str,
        delay: float = HISTORY_WRITE_DELAY,
    ) -> None:
        try:
            await asyncio.sleep(delay)
            await self._append_assistant_text_to_history(event, text)
        except Exception as e:
            logging.debug("delayed_history_write failed: %s", e)

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

        ok, chain, msg = await self._build_manual_tts_chain(event, content)
        if not ok:
            yield event.plain_result(msg)
            return

        yield event.chain_result(chain)

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
                return "文本为空"

            send_result = await self._send_manual_tts(event, content, suppress_next_llm_plain_text=True)
            if send_result == "语音已发送。":
                # 关键：工具已完成语音发送后，终止当前事件后续文本输出
                # 这样可避免 LLM 在工具调用后继续长篇打字
                event.stop_event()
                return None

            return send_result
