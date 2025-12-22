# -*- coding: utf-8 -*-
"""
TTS Emotion Router - Main Plugin Entry

按情绪路由到不同音色的 TTS 插件主入口。
"""

from __future__ import annotations

import logging
import asyncio
from typing import Dict, List, Optional

# 初始化兼容性处理（必须在其他 astrbot 导入之前）
from .core.compat import initialize_compat
initialize_compat()

# 导入 AstrBot 组件（使用兼容性包装）
from .core.compat import (
    import_astr_message_event,
    import_filter,
    import_message_components,
    import_context_and_star,
    import_astrbot_config,
    import_llm_response,
    import_result_content_type,
)

# 获取兼容的类和模块
AstrMessageEvent = import_astr_message_event()
filter = import_filter()
Record, Plain = import_message_components()
Context, Star, register = import_context_and_star()
AstrBotConfig = import_astrbot_config()
LLMResponse = import_llm_response()
ResultContentType = import_result_content_type()

# 导入核心模块
from .core.constants import (
    PLUGIN_ID,
    PLUGIN_NAME,
    PLUGIN_DESC,
    PLUGIN_VERSION,
    TEMP_DIR,
    EMOTIONS,
    EMOTION_KEYWORDS,
    AUDIO_CLEANUP_TTL_SECONDS,
    HISTORY_WRITE_DELAY,
)
from .core.session import SessionState
from .core.config import ConfigManager
from .core.marker import EmotionMarkerProcessor
from .core.tts_processor import TTSProcessor, TTSConditionChecker, TTSResultBuilder

# 导入命令处理器
from .commands.handlers import CommandHandlers

# 导入功能模块
from .emotion.classifier import HeuristicClassifier
from .tts.provider_siliconflow import SiliconFlowTTS
from .utils.audio import ensure_dir, cleanup_dir
from .utils.extract import CodeAndLinkExtractor, ProcessedText


@register(
    PLUGIN_ID,
    PLUGIN_NAME,
    PLUGIN_DESC,
    PLUGIN_VERSION,
)
class TTSEmotionRouter(Star, CommandHandlers):
    """
    TTS 情绪路由插件主类。
    
    继承自 Star 基类和 CommandHandlers Mixin，
    实现了 LLM 钩子、TTS 生成和消息处理逻辑。
    """
    
    def __init__(self, context: Context, config: Optional[dict] = None):
        super().__init__(context)
        
        # 初始化配置管理器
        self._init_config(config)
        
        # 初始化核心组件
        self._init_components()
        
        # 初始化会话状态
        self._session_state: Dict[str, SessionState] = {}
        self._inflight_sigs: set[str] = set()
        
        # 初始化临时目录
        ensure_dir(TEMP_DIR)
        # 异步清理临时目录
        asyncio.create_task(cleanup_dir(TEMP_DIR, ttl_seconds=AUDIO_CLEANUP_TTL_SECONDS))
    
    async def terminate(self):
        """插件卸载时清理资源。"""
        if hasattr(self, "tts_client"):
            await self.tts_client.close()
            logging.info("TTSEmotionRouter: tts client closed")

    def _init_config(self, config: Optional[dict]) -> None:
        """初始化配置。"""
        if isinstance(config, AstrBotConfig):
            self.config = ConfigManager(config)
        else:
            self.config = ConfigManager(config or {})
        
        # 提取常用配置到实例属性
        self.voice_map: Dict[str, str] = self.config.get_voice_map()
        self.speed_map: Dict[str, float] = self.config.get_speed_map()
        self.global_enable: bool = self.config.get_global_enable()
        self.enabled_sessions: List[str] = self.config.get_enabled_sessions()
        self.disabled_sessions: List[str] = self.config.get_disabled_sessions()
        self.prob: float = self.config.get_prob()
        self.text_limit: int = self.config.get_text_limit()
        self.cooldown: int = self.config.get_cooldown()
        self.allow_mixed: bool = self.config.get_allow_mixed()
        self.show_references: bool = self.config.get_show_references()
    
    def _init_components(self) -> None:
        """初始化各个功能组件。"""
        # 1. TTS 客户端
        api_cfg = self.config.get_api_config()
        self.tts_client = SiliconFlowTTS(
            api_cfg["url"],
            api_cfg["key"],
            api_cfg["model"],
            api_cfg["format"],
            api_cfg["speed"],
            gain=api_cfg["gain"],
            sample_rate=api_cfg["sample_rate"],
        )
        
        # 2. 情绪分类器
        emo_keywords = self.config.get_emotion_keywords()
        self.heuristic_cls = HeuristicClassifier(keywords=emo_keywords)
        
        # 3. 情绪标记处理器
        self.emo_marker_enable: bool = self.config.is_marker_enabled()
        marker_tag = self.config.get_marker_tag()
        self.marker_processor = EmotionMarkerProcessor(
            tag=marker_tag,
            enabled=self.emo_marker_enable
        )
        
        # 4. 文本提取器
        self.extractor = CodeAndLinkExtractor()
        
        # 5. 核心处理器
        self.tts_processor = TTSProcessor(
            tts_client=self.tts_client,
            voice_map=self.voice_map,
            speed_map=self.speed_map,
            heuristic_classifier=self.heuristic_cls
        )
        
        self.condition_checker = TTSConditionChecker(
            prob=self.prob,
            text_limit=self.text_limit,
            cooldown=self.cooldown,
            allow_mixed=self.allow_mixed
        )
        
        self.result_builder = TTSResultBuilder(Plain, Record)

        # 保留旧属性以保持兼容性 (部分命令处理器可能用到)
        self.tts = self.tts_client # 兼容旧代码引用 self.tts
        self.emo_marker_tag = marker_tag
        self._emo_marker_re = self.marker_processor._marker_strict_re
        self._emo_marker_re_any = self.marker_processor._marker_any_re
        self._emo_head_token_re = self.marker_processor._head_token_re
        self._emo_head_anylabel_re = self.marker_processor._head_anylabel_re
        self._emo_kw = EMOTION_KEYWORDS
    
    # ==================== 配置保存 ====================
    
    def _save_config(self) -> None:
        """保存配置。"""
        self.config.save()
        # 更新组件状态
        self.condition_checker.prob = self.config.get_prob()
        self.condition_checker.text_limit = self.config.get_text_limit()
        self.condition_checker.cooldown = self.config.get_cooldown()
        self.condition_checker.allow_mixed = self.config.get_allow_mixed()
        
        self.voice_map = self.config.get_voice_map()
        self.speed_map = self.config.get_speed_map()
        self.tts_processor.voice_map = self.voice_map
        self.tts_processor.speed_map = self.speed_map
        
        self.global_enable = self.config.get_global_enable()
        self.enabled_sessions = self.config.get_enabled_sessions()
        self.disabled_sessions = self.config.get_disabled_sessions()
        self.show_references = self.config.get_show_references()

        self.emo_marker_enable = self.config.is_marker_enabled()
        self.marker_processor.update_config(self.config.get_marker_tag(), self.emo_marker_enable)
    
    # ==================== 会话管理 ====================
    
    def _sess_id(self, event: AstrMessageEvent) -> str:
        """获取会话 ID。"""
        gid = ""
        try:
            gid = event.get_group_id()
        except Exception as e:
            logging.debug(f"TTSEmotionRouter._sess_id: failed to get group_id: {e}")
            gid = ""
        if gid:
            return f"group_{gid}"
        return f"user_{event.get_sender_id()}"
    
    def _is_session_enabled(self, sid: str) -> bool:
        """检查会话是否启用 TTS。"""
        if self.global_enable:
            return sid not in self.disabled_sessions
        return sid in self.enabled_sessions

    def _get_session_state(self, sid: str) -> SessionState:
        return self._session_state.setdefault(sid, SessionState())
    
    # ==================== 文本处理代理 ====================
    # 为了保持与 CommandHandlers 的兼容性，保留这些方法
    
    def _normalize_text(self, text: str) -> str:
        return self.marker_processor.normalize_text(text)
    
    def _strip_emo_head_many(self, text: str) -> tuple[str, Optional[str]]:
        return self.marker_processor.strip_head_many(text)
        
    def _strip_any_visible_markers(self, text: str) -> str:
        return self.marker_processor.strip_all_visible_markers(text)

    def _normalize_audio_path(self, path):
         return self.tts_processor.normalize_audio_path(path)

    def _pick_voice_for_emotion(self, emotion: str):
        return self.tts_processor.pick_voice_for_emotion(emotion)

    # ==================== LLM 钩子 ====================
    
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, request):
        """在系统提示中注入情绪标记指令。"""
        if not self.emo_marker_enable:
            return
        
        try:
            sp = getattr(request, "system_prompt", "") or ""
            pp = getattr(request, "prompt", "") or ""
            
            if not self.marker_processor.is_marker_present(sp, pp):
                instr = self.marker_processor.build_injection_instruction()
                try:
                    request.system_prompt = (instr + "\n" + sp).strip()
                    logging.info("TTSEmotionRouter: injected emotion marker instruction")
                except Exception as e:
                    logging.warning(f"TTSEmotionRouter.on_llm_request: failed to inject prompt: {e}")
        except Exception as e:
            logging.error(f"TTSEmotionRouter.on_llm_request: unexpected error: {e}")
    
    @filter.on_llm_response(priority=1)
    async def on_llm_response(self, event: AstrMessageEvent, response: LLMResponse):
        """解析并清理 LLM 响应中的情绪标记。"""
        if not self.emo_marker_enable:
            return
        
        label: Optional[str] = None
        cached_text: Optional[str] = None
        
        # 1) 从 completion_text 提取并清理
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
            logging.warning(f"TTSEmotionRouter.on_llm_response: failed to strip markers from completion_text: {e}")
        
        # 2) 从 result_chain 首个 Plain 再尝试一次
        try:
            rc = getattr(response, "result_chain", None)
            if rc and hasattr(rc, "chain") and rc.chain:
                new_chain = []
                cleaned_once = False
                for comp in rc.chain:
                    if (
                        not cleaned_once
                        and isinstance(comp, Plain)
                        and getattr(comp, "text", None)
                    ):
                        t0 = self._normalize_text(comp.text)
                        t, l2 = self._strip_emo_head_many(t0)
                        if l2 in EMOTIONS and label is None:
                            label = l2
                        if t:
                            new_chain.append(Plain(text=t))
                            try:
                                if t and not getattr(response, "_completion_text", None):
                                    setattr(response, "_completion_text", t)
                            except Exception:
                                pass
                            cached_text = t or cached_text
                        cleaned_once = True
                    else:
                        new_chain.append(comp)
                rc.chain = new_chain
        except Exception as e:
            logging.warning(f"TTSEmotionRouter.on_llm_response: failed to strip markers from result_chain: {e}")
        
        # 3) 记录到 session
        try:
            sid = self._sess_id(event)
            st = self._get_session_state(sid)
            if label in EMOTIONS:
                st.pending_emotion = label
            if cached_text and cached_text.strip():
                st.set_assistant_text(cached_text.strip())
        except Exception as e:
            logging.error(f"TTSEmotionRouter.on_llm_response: failed to update session state: {e}")
        
        # 4) 写入会话历史
        try:
            if cached_text and cached_text.strip():
                ok = await self._append_assistant_text_to_history(event, cached_text.strip())
                if not ok:
                    asyncio.create_task(self._delayed_history_write(event, cached_text.strip(), delay=HISTORY_WRITE_DELAY))
        except Exception as e:
            logging.error(f"TTSEmotionRouter.on_llm_response: failed to append history: {e}")
    
    @filter.on_decorating_result(priority=999)
    async def _final_strip_markers(self, event: AstrMessageEvent):
        """最终装饰阶段：兜底去除情绪标记泄露。"""
        try:
            if not self.emo_marker_enable:
                return
            result = event.get_result()
            if not result or not hasattr(result, 'chain'):
                return
            changed = False
            for comp in list(result.chain):
                if isinstance(comp, Plain) and getattr(comp, 'text', None):
                    new_txt = self._strip_any_visible_markers(comp.text)
                    if new_txt != comp.text:
                        comp.text = new_txt
                        changed = True
            if changed:
                logging.debug("TTSEmotionRouter: final marker cleanup applied")
        except Exception as e:
            logging.error(f"TTSEmotionRouter._final_strip_markers: unexpected error: {e}")

    # ==================== 核心 TTS 处理钩子 ====================
    
    @filter.on_decorating_result(priority=-1000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """核心 TTS 处理逻辑。"""
        # 声明继续传播
        try:
            event.continue_event()
        except Exception:
            pass
        
        # 若已为 STOP，切回 CONTINUE
        try:
            if event.is_stopped():
                logging.info("TTSEmotionRouter: detected STOP at entry, forcing CONTINUE")
                event.continue_event()
        except Exception:
            pass
        
        # 1. 基础检查：是否为 LLM 响应
        try:
            result = event.get_result()
            if not result:
                return
                
            is_llm_response = False
            try:
                is_llm_response = result.is_llm_result()
            except Exception:
                is_llm_response = (getattr(result, "result_content_type", None) == ResultContentType.LLM_RESULT)
            
            if not is_llm_response:
                return # 静默跳过非 LLM 消息
                
            if not result.chain:
                return
        except Exception as e:
            logging.warning(f"TTS: error checking response type: {e}")
            return
            
        # 2. 会话开关检查
        sid = self._sess_id(event)
        if not self._is_session_enabled(sid):
            logging.info("TTS skip: session disabled (%s)", sid)
            return

        # 3. 强制清理情绪标记
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
            logging.warning(f"TTSEmotionRouter: failed to strip markers: {e}")

        # 4. 提取纯文本
        text_parts = [
            c.text.strip()
            for c in result.chain
            if isinstance(c, Plain) and c.text.strip()
        ]
        if not text_parts:
            return
        text = " ".join(text_parts)
        
        # 5. 文本预处理 (代码/链接提取)
        orig_text = text
        text = self._normalize_text(text)
        text, _ = self._strip_emo_head_many(text)
        
        processed: ProcessedText = self.extractor.process_text(text)
        tts_text = processed.speak_text
        clean_text = processed.clean_text
        links = processed.links
        codes = processed.codes
        
        # 构建发送文本 (包含参考文献)
        send_text = clean_text.strip()
        if self.show_references:
            if links:
                send_text += "\n\n参考文献:\n" + "\n".join(f"{i+1}. {link}" for i, link in enumerate(links))
            if codes:
                send_text += "\n\n参考代码:\n" + "\n".join(codes)

        # 6. 如果无可读文本，直接返回处理后的文本
        if not tts_text.strip():
            result.chain = [Plain(text=send_text)]
            return

        # 7. 条件检查 (Condition Checker)
        st = self._get_session_state(sid)
        has_non_plain = any(not isinstance(c, Plain) for c in result.chain)
        
        check_res = self.condition_checker.check_all(tts_text, st, has_non_plain)
        if not check_res.passed:
            logging.info(f"TTS skip: {check_res.reason}")
            # 如果是因为混合内容跳过，至少要把处理过的文本（如去除了代码块）放回去
            if "mixed content" in check_res.reason:
                 result.chain = [Plain(text=send_text)] + [c for c in result.chain if not isinstance(c, Plain)]
            return

        # 8. 防重检查
        sig = f"{sid}:{hash(tts_text[:50])}"
        if sig in self._inflight_sigs:
            logging.info("TTS skip: duplicate request in flight")
            return
        self._inflight_sigs.add(sig)

        try:
            # 9. 执行 TTS 处理 (Core Processor)
            proc_res = await self.tts_processor.process(tts_text, st)
            
            if proc_res.success and proc_res.audio_path:
                # 10. 构建结果 (Result Builder)
                norm_path = self.tts_processor.normalize_audio_path(proc_res.audio_path)
                
                # 检查会话级文字语音同显配置
                session_text_voice = st.text_voice_enabled
                effective_text_voice = session_text_voice if session_text_voice is not None else self.config.get_text_voice_default()
                
                result.chain = self.result_builder.build(
                    original_chain=result.chain,
                    audio_path=norm_path,
                    send_text=send_text,
                    text_voice_enabled=effective_text_voice
                )
                
                logging.info(f"TTS: success, audio={norm_path}")
                
                # 缓存文本用于历史记录
                if send_text.strip():
                    st.set_assistant_text(send_text.strip())
            else:
                # 失败则回退到纯文本
                logging.error(f"TTS failed: {proc_res.error}")
                result.chain = [Plain(text=send_text)]

        finally:
            self._inflight_sigs.discard(sig)

    # ==================== 历史记录辅助方法 ====================
    
    async def _ensure_history_saved(self, event: AstrMessageEvent) -> None:
        """确保助手文本已保存到历史记录。"""
        try:
            sid = self._sess_id(event)
            st = self._session_state.get(sid)
            if not st or not st.assistant_text:
                return
            
            text = st.assistant_text
            st.assistant_text = None
            
            await self._append_assistant_text_to_history(event, text)
        except Exception as e:
            logging.debug("TTSEmotionRouter: _ensure_history_saved error: %s", e)
    
    async def _append_assistant_text_to_history(self, event: AstrMessageEvent, text: str) -> bool:
        """将助手文本追加到会话历史。"""
        try:
            if not text or not text.strip():
                return False
            
            provider = getattr(self.context, "llm_provider", None)
            try:
                if not provider:
                    provider = self.context.get_llm_provider()
            except Exception:
                pass
            
            if not provider:
                return False
            
            sid = self._sess_id(event)
            
            try:
                if hasattr(provider, "append_assistant_response"):
                    await provider.append_assistant_response(sid, text)
                    return True
                elif hasattr(provider, "add_message"):
                    await provider.add_message(sid, "assistant", text)
                    return True
            except Exception:
                pass
            
            return False
        except Exception:
            return False
    
    async def _delayed_history_write(self, event: AstrMessageEvent, text: str, delay: float = HISTORY_WRITE_DELAY) -> None:
        """延迟写入历史记录。"""
        try:
            await asyncio.sleep(delay)
            await self._append_assistant_text_to_history(event, text)
        except Exception as e:
            logging.debug(f"TTSEmotionRouter._delayed_history_write failed: {e}")

    # ==================== 消息发送后钩子 ====================
    
    if hasattr(filter, "after_message_sent"):
        @filter.after_message_sent(priority=-1000)
        async def after_message_sent(self, event: AstrMessageEvent):
            """消息发送后的处理。"""
            try:
                try:
                    event.continue_event()
                except Exception:
                    pass
                
                # 确保上游处理结果继续
                res = event.get_result()
                if res is not None and hasattr(res, "continue_event"):
                    try:
                        res.continue_event()
                    except Exception:
                        pass
                
                result = event.get_result()
                if not result or not getattr(result, "chain", None):
                    return
                
                # 兜底：确保语音消息时可读文本写入历史
                try:
                    if any(isinstance(c, Record) for c in result.chain):
                        await self._ensure_history_saved(event)
                except Exception as e:
                    logging.warning(f"TTSEmotionRouter.after_message_sent: ensure_history_saved failed: {e}")
                
            except Exception as e:
                logging.error(f"TTSEmotionRouter.after_message_sent: unexpected error: {e}")
    else:
        async def after_message_sent(self, event: AstrMessageEvent):
            return

    # ==================== 命令注册 ====================
    # (委托给 CommandHandlers Mixin，但需要确保方法名对应)
    
    @filter.command("tts_marker_on", priority=1)
    async def tts_marker_on(self, event: AstrMessageEvent):
        result = await self.cmd_tts_marker_on(event)
        yield event.plain_result(result)
    
    @filter.command("tts_marker_off", priority=1)
    async def tts_marker_off(self, event: AstrMessageEvent):
        result = await self.cmd_tts_marker_off(event)
        yield event.plain_result(result)
    
    @filter.command("tts_emote", priority=1)
    async def tts_emote(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        result = await self.cmd_tts_emote(event, value)
        yield event.plain_result(result)
    
    @filter.command("tts_global_on", priority=1)
    async def tts_global_on(self, event: AstrMessageEvent):
        result = await self.cmd_tts_global_on(event)
        yield event.plain_result(result)
    
    @filter.command("tts_global_off", priority=1)
    async def tts_global_off(self, event: AstrMessageEvent):
        result = await self.cmd_tts_global_off(event)
        yield event.plain_result(result)
    
    @filter.command("tts_on", priority=1)
    async def tts_on(self, event: AstrMessageEvent):
        result = await self.cmd_tts_on(event)
        yield event.plain_result(result)
    
    @filter.command("tts_off", priority=1)
    async def tts_off(self, event: AstrMessageEvent):
        result = await self.cmd_tts_off(event)
        yield event.plain_result(result)
    
    @filter.command("tts_prob", priority=1)
    async def tts_prob(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        result = await self.cmd_tts_prob(event, value)
        yield event.plain_result(result)
    
    @filter.command("tts_limit", priority=1)
    async def tts_limit(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        result = await self.cmd_tts_limit(event, value)
        yield event.plain_result(result)
    
    @filter.command("tts_cooldown", priority=1)
    async def tts_cooldown(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        result = await self.cmd_tts_cooldown(event, value)
        yield event.plain_result(result)
    
    @filter.command("tts_test", priority=1)
    async def tts_test(self, event: AstrMessageEvent, *, text: Optional[str] = None):
        async for result in self.cmd_tts_test(event, text):
            if isinstance(result, tuple) and result[0] == "__AUDIO__":
                try:
                    yield event.chain_result([Record(file=result[1])])
                except Exception as e:
                    yield event.plain_result(f"❌ 音频发送失败: {e}")
            else:
                yield event.plain_result(result)
    
    @filter.command("tts_debug", priority=1)
    async def tts_debug(self, event: AstrMessageEvent):
        result = await self.cmd_tts_debug(event)
        yield event.plain_result(result)
    
    @filter.command("tts_gain", priority=1)
    async def tts_gain(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        result = await self.cmd_tts_gain(event, value)
        yield event.plain_result(result)
    
    @filter.command("tts_status", priority=1)
    async def tts_status(self, event: AstrMessageEvent):
        result = await self.cmd_tts_status(event)
        yield event.plain_result(result)
    
    @filter.command("tts_mixed_on", priority=1)
    async def tts_mixed_on(self, event: AstrMessageEvent):
        result = await self.cmd_tts_mixed_on(event)
        yield event.plain_result(result)
    
    @filter.command("tts_mixed_off", priority=1)
    async def tts_mixed_off(self, event: AstrMessageEvent):
        result = await self.cmd_tts_mixed_off(event)
        yield event.plain_result(result)
    
    @filter.command("tts_text_voice_on", priority=1)
    async def tts_text_voice_on(self, event: AstrMessageEvent):
        result = await self.cmd_tts_text_voice_on(event)
        yield event.plain_result(result)
    
    @filter.command("tts_text_voice_off", priority=1)
    async def tts_text_voice_off(self, event: AstrMessageEvent):
        result = await self.cmd_tts_text_voice_off(event)
        yield event.plain_result(result)
    
    @filter.command("tts_text_voice_reset", priority=1)
    async def tts_text_voice_reset(self, event: AstrMessageEvent):
        result = await self.cmd_tts_text_voice_reset(event)
        yield event.plain_result(result)
    
    @filter.command("tts_check_refs", priority=1)
    async def tts_check_refs(self, event: AstrMessageEvent):
        result = await self.cmd_tts_check_refs(event)
        yield event.plain_result(result)
    
    @filter.command("tts_refs_on", priority=1)
    async def tts_refs_on(self, event: AstrMessageEvent):
        result = await self.cmd_tts_refs_on(event)
        yield event.plain_result(result)
    
    @filter.command("tts_refs_off", priority=1)
    async def tts_refs_off(self, event: AstrMessageEvent):
        result = await self.cmd_tts_refs_off(event)
        yield event.plain_result(result)
