# -*- coding: utf-8 -*-
"""
TTS Emotion Router - Main Plugin Entry

按情绪路由到不同音色的 TTS 插件主入口。
"""

from __future__ import annotations

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Set

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
    import_message_chain,
)

# 获取兼容的类和模块
AstrMessageEvent = import_astr_message_event()
filter = import_filter()
Record, Plain = import_message_components()
Context, Star, register = import_context_and_star()
AstrBotConfig = import_astrbot_config()
LLMResponse = import_llm_response()
ResultContentType = import_result_content_type()
MessageChain = import_message_chain()

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

# 会话状态清理常量
SESSION_CLEANUP_INTERVAL = 3600  # 每小时检查一次
SESSION_MAX_IDLE_TIME = 86400  # 24小时无活动则清理
SESSION_MAX_COUNT = 10000  # 最大会话数量
from .core.session import SessionState
from .core.config import ConfigManager
from .core.marker import EmotionMarkerProcessor
from .core.tts_processor import TTSProcessor, TTSConditionChecker, TTSResultBuilder
from .core.segmented_tts import SegmentedTTSProcessor, get_audio_duration, INTERVAL_MODE_FIXED, INTERVAL_MODE_ADAPTIVE
from .core.text_splitter import TextSplitter

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
        
        # 后台任务引用（用于在卸载时取消）
        self._background_tasks: List[asyncio.Task] = []
        self._cleanup_task_started: bool = False
        
        # 初始化临时目录（同步，因为在初始化阶段）
        ensure_dir(TEMP_DIR)
    
    async def terminate(self):
        """插件卸载时清理资源。"""
        # 取消所有后台任务
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._background_tasks.clear()
        
        # 关闭 TTS 客户端
        if hasattr(self, "tts_client"):
            await self.tts_client.close()
            logging.info("TTSEmotionRouter: tts client closed")
        
        # 清理会话状态
        self._session_state.clear()
        logging.info("TTSEmotionRouter: session state cleared")

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
        # 使用 UMO（统一消息来源）替代原来的 session_id
        self.enabled_umos: List[str] = self.config.get_enabled_umos()
        self.disabled_umos: List[str] = self.config.get_disabled_umos()
        # 保留旧属性名用于兼容，指向新属性
        self.enabled_sessions: List[str] = self.enabled_umos
        self.disabled_sessions: List[str] = self.disabled_umos
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
            text_min_limit=self.config.get_text_min_limit(),
            cooldown=self.cooldown,
            allow_mixed=self.allow_mixed
        )
        
        self.result_builder = TTSResultBuilder(Plain, Record)

        # 6. 分段 TTS 处理器
        self._init_segmented_tts()

        # 保留旧属性以保持兼容性 (部分命令处理器可能用到)
        self.tts = self.tts_client # 兼容旧代码引用 self.tts
        self.emo_marker_tag = marker_tag
        self._emo_marker_re = self.marker_processor._marker_strict_re
        self._emo_marker_re_any = self.marker_processor._marker_any_re
        self._emo_head_token_re = self.marker_processor._head_token_re
        self._emo_head_anylabel_re = self.marker_processor._head_anylabel_re
        self._emo_kw = EMOTION_KEYWORDS
    
    # ==================== 配置保存 ====================
    
    async def _save_config_async(self) -> None:
        """异步保存配置（推荐使用）。"""
        await self.config.save_async()
        self._update_components_from_config()
    
    def _save_config(self) -> None:
        """同步保存配置（已废弃，仅用于兼容）。"""
        # 警告：此方法会阻塞事件循环，请使用 _save_config_async
        logging.warning("TTSEmotionRouter: _save_config() is deprecated, use _save_config_async() instead")
        self.config.save()
        self._update_components_from_config()
    
    def _update_components_from_config(self) -> None:
        """从配置更新组件状态。"""
        # 更新组件状态
        self.condition_checker.prob = self.config.get_prob()
        self.condition_checker.text_limit = self.config.get_text_limit()
        self.condition_checker.text_min_limit = self.config.get_text_min_limit()
        self.condition_checker.cooldown = self.config.get_cooldown()
        self.condition_checker.allow_mixed = self.config.get_allow_mixed()
        
        self.voice_map = self.config.get_voice_map()
        self.speed_map = self.config.get_speed_map()
        self.tts_processor.voice_map = self.voice_map
        self.tts_processor.speed_map = self.speed_map
        
        self.global_enable = self.config.get_global_enable()
        # 使用 UMO 替代 session
        self.enabled_umos = self.config.get_enabled_umos()
        self.disabled_umos = self.config.get_disabled_umos()
        # 保留旧属性名用于兼容
        self.enabled_sessions = self.enabled_umos
        self.disabled_sessions = self.disabled_umos
        self.show_references = self.config.get_show_references()

        self.emo_marker_enable = self.config.is_marker_enabled()
        self.marker_processor.update_config(self.config.get_marker_tag(), self.emo_marker_enable)
        
        # 更新分段 TTS 处理器
        self._init_segmented_tts()
    
    def _init_segmented_tts(self) -> None:
        """初始化分段 TTS 处理器。"""
        # 获取分段配置
        seg_cfg = self.config.get_segmented_tts_config()
        
        # 创建文本分段器
        self.text_splitter = TextSplitter(
            split_pattern=self.config.get_segmented_tts_split_pattern(),
            smart_mode=True,
            max_segments=self.config.get_segmented_tts_max_segments(),
            min_segment_length=2,
        )
        
        # 创建分段 TTS 处理器
        interval_mode = self.config.get_segmented_tts_interval_mode()
        self.segmented_tts_processor = SegmentedTTSProcessor(
            tts_processor=self.tts_processor,
            splitter=self.text_splitter,
            interval_mode=interval_mode,
            fixed_interval=self.config.get_segmented_tts_fixed_interval(),
            adaptive_buffer=self.config.get_segmented_tts_adaptive_buffer(),
            max_segments=self.config.get_segmented_tts_max_segments(),
        )
        
        # 缓存配置
        self.segmented_tts_enabled = self.config.is_segmented_tts_enabled()
        self.segmented_tts_min_chars = self.config.get_segmented_tts_min_segment_chars()
    
    # ==================== 会话管理（UMO 版本） ====================
    
    def _get_umo(self, event: AstrMessageEvent) -> str:
        """
        获取 UMO（统一消息来源）标识。
        
        UMO 是 AstrBot 的标准会话标识格式，可通过 /sid 命令查看。
        
        Args:
            event: 消息事件对象
            
        Returns:
            str: UMO 标识字符串
        """
        # 直接使用 AstrBot 的统一消息来源标识
        umo = getattr(event, "unified_msg_origin", None)
        if umo:
            logging.info(f"TTSEmotionRouter._get_umo: unified_msg_origin={umo}")
            return str(umo)
        
        # 回退方案：如果 unified_msg_origin 不存在，使用旧的拼接方式
        gid = ""
        try:
            gid = event.get_group_id()
        except Exception as e:
            logging.debug(f"TTSEmotionRouter._get_umo: failed to get group_id: {e}")
            gid = ""
        
        # 确保 gid 是真正有效的群组 ID（非空、非 None 字符串）
        if gid and gid not in ("", "None", "null", "0"):
            sid = f"group_{gid}"
        else:
            sid = f"user_{event.get_sender_id()}"
        
        logging.info(f"TTSEmotionRouter._get_umo: fallback mode, gid={gid!r}, sender={event.get_sender_id()}, result={sid}")
        return sid
    
    def _sess_id(self, event: AstrMessageEvent) -> str:
        """获取会话 ID（已废弃，请使用 _get_umo）。"""
        return self._get_umo(event)
    
    def _is_umo_enabled(self, umo: str) -> bool:
        """
        检查 UMO 是否启用 TTS。
        
        Args:
            umo: 统一消息来源标识
            
        Returns:
            bool: 如果启用返回 True
        """
        if self.global_enable:
            # 黑名单模式：默认开启，在黑名单中则关闭
            return umo not in self.disabled_umos
        # 白名单模式：默认关闭，在白名单中则开启
        return umo in self.enabled_umos
    
    def _is_session_enabled(self, sid: str) -> bool:
        """检查会话是否启用 TTS（已废弃，请使用 _is_umo_enabled）。"""
        return self._is_umo_enabled(sid)

    def _get_session_state(self, sid: str) -> SessionState:
        return self._session_state.setdefault(sid, SessionState())
    
    async def _start_background_tasks(self) -> None:
        """启动后台任务（在首次处理消息时调用）。"""
        if self._cleanup_task_started:
            return
        self._cleanup_task_started = True
        
        # 启动临时文件清理任务
        audio_cleanup_task = asyncio.create_task(
            self._periodic_audio_cleanup(),
            name="tts_audio_cleanup"
        )
        self._background_tasks.append(audio_cleanup_task)
        
        # 启动会话状态清理任务
        session_cleanup_task = asyncio.create_task(
            self._periodic_session_cleanup(),
            name="tts_session_cleanup"
        )
        self._background_tasks.append(session_cleanup_task)
        
        logging.info("TTSEmotionRouter: background tasks started")
    
    async def _periodic_audio_cleanup(self) -> None:
        """定期清理临时音频文件。"""
        try:
            while True:
                await cleanup_dir(TEMP_DIR, ttl_seconds=AUDIO_CLEANUP_TTL_SECONDS)
                await asyncio.sleep(AUDIO_CLEANUP_TTL_SECONDS // 2)  # 每隔一半TTL时间清理一次
        except asyncio.CancelledError:
            logging.debug("TTSEmotionRouter: audio cleanup task cancelled")
            raise
        except Exception as e:
            logging.error(f"TTSEmotionRouter: audio cleanup error: {e}")
    
    async def _periodic_session_cleanup(self) -> None:
        """定期清理过期会话状态，防止内存泄漏。"""
        try:
            while True:
                await asyncio.sleep(SESSION_CLEANUP_INTERVAL)
                await self._cleanup_stale_sessions()
        except asyncio.CancelledError:
            logging.debug("TTSEmotionRouter: session cleanup task cancelled")
            raise
        except Exception as e:
            logging.error(f"TTSEmotionRouter: session cleanup error: {e}")
    
    async def _cleanup_stale_sessions(self) -> None:
        """清理过期的会话状态。"""
        import time
        now = time.time()
        stale_sessions = []
        
        for sid, state in self._session_state.items():
            # 检查是否超过最大空闲时间
            if now - state.last_ts > SESSION_MAX_IDLE_TIME:
                stale_sessions.append(sid)
        
        # 如果会话数量超过限制，额外清理最久未使用的会话
        if len(self._session_state) > SESSION_MAX_COUNT:
            # 按最后活跃时间排序
            sorted_sessions = sorted(
                self._session_state.items(),
                key=lambda x: x[1].last_ts
            )
            # 清理超出限制的会话（保留最近活跃的）
            excess_count = len(self._session_state) - SESSION_MAX_COUNT
            for sid, _ in sorted_sessions[:excess_count]:
                if sid not in stale_sessions:
                    stale_sessions.append(sid)
        
        # 执行清理
        for sid in stale_sessions:
            del self._session_state[sid]
        
        if stale_sessions:
            logging.info(f"TTSEmotionRouter: cleaned up {len(stale_sessions)} stale sessions, remaining: {len(self._session_state)}")
    
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
        logging.info("TTSEmotionRouter.on_decorating_result: ENTRY")
        
        # 启动后台任务（仅首次）
        await self._start_background_tasks()
        
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
                logging.info("TTS skip: no result object")
                return
                
            is_llm_response = False
            try:
                is_llm_response = result.is_llm_result()
            except Exception:
                is_llm_response = (getattr(result, "result_content_type", None) == ResultContentType.LLM_RESULT)
            
            logging.info(f"TTS check: is_llm_response={is_llm_response}, chain_len={len(result.chain) if result.chain else 0}")
            
            if not is_llm_response:
                logging.info("TTS skip: not LLM response")
                return # 静默跳过非 LLM 消息
                
            if not result.chain:
                logging.info("TTS skip: empty chain")
                return
        except Exception as e:
            logging.warning(f"TTS: error checking response type: {e}")
            return
            
        # 2. 会话开关检查（使用 UMO）
        umo = self._get_umo(event)
        logging.info(f"TTS check: umo={umo}, global_enable={self.global_enable}, enabled_count={len(self.enabled_umos)}, disabled_count={len(self.disabled_umos)}")
        if not self._is_umo_enabled(umo):
            logging.info("TTS skip: UMO disabled (%s)", umo)
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

        # 3.5 提取并分离 Image 组件（兼容 meme_manager 等插件）
        # Image 组件将被单独发送，不影响 TTS 生成
        image_components = []
        non_image_chain = []
        for comp in result.chain:
            c_type = type(comp).__name__
            if c_type == "Image":
                image_components.append(comp)
            else:
                non_image_chain.append(comp)
        
        # 如果有 Image 组件，先单独发送
        if image_components:
            logging.info(f"TTS: detected {len(image_components)} Image component(s), sending separately")
            try:
                # 使用 event.send() 单独发送图片
                for img_comp in image_components:
                    await event.send(MessageChain(chain=[img_comp]))
                    logging.info("TTS: Image component sent successfully")
            except Exception as e:
                logging.warning(f"TTS: failed to send Image component separately: {e}")
                # 如果发送失败，将图片放回结果链
                non_image_chain.extend(image_components)
        
        # 更新 result.chain 为不含 Image 的链
        result.chain = non_image_chain

        # 4. 提取纯文本
        text_parts = [
            c.text.strip()
            for c in result.chain
            if isinstance(c, Plain) and c.text.strip()
        ]
        if not text_parts:
            # 如果已经发送了图片但没有文本，清空结果链
            if image_components:
                result.chain = []
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
        st = self._get_session_state(umo)
        
        # 混合内容检查：允许 Plain, At, Reply, Face 等组件
        # 注意：Image 已被提取并单独发送，所以这里不再需要包含 Image
        # 使用类名字符串匹配，避免不同版本的导入问题
        ALLOWED_COMPONENTS = {"Plain", "At", "Reply", "Face"}
        has_non_plain = False
        non_plain_types = []
        for c in result.chain:
            c_type = type(c).__name__
            if c_type not in ALLOWED_COMPONENTS:
                has_non_plain = True
                non_plain_types.append(c_type)
        
        logging.info(f"TTS check: tts_text_len={len(tts_text)}, has_non_plain={has_non_plain}, non_plain_types={non_plain_types}")
        
        check_res = self.condition_checker.check_all(tts_text, st, has_non_plain)
        if not check_res.passed:
            logging.info(f"TTS skip: {check_res.reason}")
            # 如果是因为混合内容跳过，至少要把处理过的文本（如去除了代码块）放回去
            if "mixed content" in check_res.reason:
                 result.chain = [Plain(text=send_text)] + [c for c in result.chain if not isinstance(c, Plain)]
            return

        # 8. 防重检查
        sig = f"{umo}:{hash(tts_text[:50])}"
        if sig in self._inflight_sigs:
            logging.info("TTS skip: duplicate request in flight")
            return
        self._inflight_sigs.add(sig)

        try:
            # 9. 判断是否使用分段 TTS
            use_segmented = (
                self.segmented_tts_enabled 
                and self.segmented_tts_processor.should_use_segmented(tts_text, self.segmented_tts_min_chars)
            )
            
            if use_segmented:
                # 使用分段 TTS 处理
                logging.info(f"TTS: using segmented mode for text: {tts_text[:50]}...")
                
                # 定义发送函数
                async def send_audio(audio_path: Path) -> bool:
                    try:
                        await event.send(MessageChain(chain=[Record(file=str(audio_path))]))
                        return True
                    except Exception as e:
                        logging.error(f"TTS: failed to send audio segment: {e}")
                        return False
                
                # 执行分段处理并发送
                seg_result = await self.segmented_tts_processor.process_and_send(
                    tts_text, st, send_audio
                )
                
                if seg_result.success:
                    logging.info(f"TTS: segmented mode success, sent {len(seg_result.successful_segments)} segment(s)")
                    # 分段模式下，所有语音已发送，清空结果链
                    # 如果有链接或代码，发送参考文献
                    if links or codes:
                        ref_parts = []
                        if links:
                            ref_parts.extend(links)
                        if codes:
                            ref_parts.extend(codes)
                        ref_text = "\n".join(ref_parts)
                        result.chain = [Plain(text=ref_text)]
                    else:
                        result.chain = []
                    
                    # 缓存文本用于历史记录
                    if send_text.strip():
                        st.set_assistant_text(send_text.strip())
                else:
                    # 分段失败，回退到普通模式
                    logging.warning(f"TTS: segmented mode failed ({seg_result.error}), falling back to normal mode")
                    use_segmented = False
            
            if not use_segmented:
                # 10. 执行普通 TTS 处理 (Core Processor)
                logging.info(f"TTS: starting TTS processing for text: {tts_text[:50]}...")
                proc_res = await self.tts_processor.process(tts_text, st)
                logging.info(f"TTS: process result: success={proc_res.success}, audio_path={proc_res.audio_path}, error={proc_res.error}")
            
                if proc_res.success and proc_res.audio_path:
                    # 11. 构建结果 (Result Builder)
                    norm_path = self.tts_processor.normalize_audio_path(proc_res.audio_path)

                    # 检查 UMO 级别的文字+语音同显配置
                    # 优先级: 会话状态 > UMO 配置 > 全局默认
                    session_text_voice = st.text_voice_enabled
                    if session_text_voice is not None:
                        # 会话状态有明确设置，优先使用
                        effective_text_voice = session_text_voice
                    elif self.config.is_text_voice_enabled_for_umo(umo):
                        # UMO 在文字+语音同显列表中
                        effective_text_voice = True
                    else:
                        # 使用全局默认值
                        effective_text_voice = self.config.get_text_voice_default()

                    # 如果有链接或代码，强制发送文本（方便用户复制）
                    # 只发送链接和代码，不重复发送整段文字
                    if links or codes:
                        ref_parts = []
                        if links:
                            ref_parts.extend(links)
                        if codes:
                            ref_parts.extend(codes)
                        ref_text = "\n".join(ref_parts)
                        result.chain = self.result_builder.build(
                            original_chain=result.chain,
                            audio_path=norm_path,
                            send_text=ref_text,
                            text_voice_enabled=True
                        )
                    else:
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
    
    # ==================== 分段 TTS 命令 ====================
    
    @filter.command("tts_segment_on", priority=1)
    async def tts_segment_on(self, event: AstrMessageEvent):
        result = await self.cmd_tts_segment_on(event)
        yield event.plain_result(result)
    
    @filter.command("tts_segment_off", priority=1)
    async def tts_segment_off(self, event: AstrMessageEvent):
        result = await self.cmd_tts_segment_off(event)
        yield event.plain_result(result)
    
    @filter.command("tts_segment_mode", priority=1)
    async def tts_segment_mode(self, event: AstrMessageEvent, *, mode: Optional[str] = None):
        result = await self.cmd_tts_segment_mode(event, mode)
        yield event.plain_result(result)
    
    @filter.command("tts_segment_interval", priority=1)
    async def tts_segment_interval(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        result = await self.cmd_tts_segment_interval(event, value)
        yield event.plain_result(result)
    
    @filter.command("tts_segment_status", priority=1)
    async def tts_segment_status(self, event: AstrMessageEvent):
        result = await self.cmd_tts_segment_status(event)
        yield event.plain_result(result)
