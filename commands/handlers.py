# -*- coding: utf-8 -*-
"""
TTS Emotion Router - Command Handlers

命令处理模块，包含所有 tts_* 命令的实现。
注意：此模块中的类是一个 Mixin，需要与主插件类一起使用。
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from ..core.session import SessionState
    from ..core.config import ConfigManager

logger = logging.getLogger(__name__)


class CommandHandlers:
    """
    命令处理器 Mixin 类。
    
    包含所有 tts_* 命令的实现逻辑。
    这是一个 Mixin 类，需要与主插件类组合使用。
    
    依赖的属性（由主插件类提供）：
        - config: ConfigManager
        - emo_marker_enable: bool
        - global_enable: bool
        - enabled_sessions: List[str]
        - disabled_sessions: List[str]
        - prob: float
        - text_limit: int
        - cooldown: int
        - allow_mixed: bool
        - show_references: bool
        - tts: SiliconFlowTTS
        - voice_map: Dict[str, str]
        - speed_map: Dict[str, float]
        - _session_state: Dict[str, SessionState]
        - marker_processor: EmotionMarkerProcessor
    """
    
    # ==================== 情绪标记命令 ====================
    
    async def cmd_tts_marker_on(self, event) -> str:
        """开启情绪隐藏标记。"""
        try:
            self.emo_marker_enable = True  # type: ignore
            await self.config.set_marker_enable_async(True)  # type: ignore
            return "情绪隐藏标记：开启"
        except Exception as e:
            logger.error(f"cmd_tts_marker_on failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    async def cmd_tts_marker_off(self, event) -> str:
        """关闭情绪隐藏标记。"""
        try:
            self.emo_marker_enable = False  # type: ignore
            await self.config.set_marker_enable_async(False)  # type: ignore
            return "情绪隐藏标记：关闭"
        except Exception as e:
            logger.error(f"cmd_tts_marker_off failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    async def cmd_tts_emote(self, event, value: Optional[str] = None) -> str:
        """手动指定下一条消息的情绪。"""
        from ..core.constants import EMOTIONS
        
        try:
            label = (value or "").strip().lower()
            assert label in EMOTIONS
            sid = self._sess_id(event)  # type: ignore
            st = self._session_state.setdefault(sid, self._create_session_state())  # type: ignore
            st.pending_emotion = label
            return f"已设置：下一条消息按情绪 {label} 路由"
        except AssertionError:
            return "用法：tts_emote <happy|sad|angry|neutral>"
        except Exception as e:
            logger.error(f"cmd_tts_emote failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    # ==================== 全局开关命令 ====================
    
    async def cmd_tts_global_on(self, event) -> str:
        """开启全局 TTS（黑名单模式）。"""
        try:
            self.global_enable = True  # type: ignore
            await self.config.set_global_enable_async(True)  # type: ignore
            return "TTS 全局：开启（黑名单模式）"
        except Exception as e:
            logger.error(f"cmd_tts_global_on failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    async def cmd_tts_global_off(self, event) -> str:
        """关闭全局 TTS（白名单模式）。"""
        try:
            self.global_enable = False  # type: ignore
            await self.config.set_global_enable_async(False)  # type: ignore
            return "TTS 全局：关闭（白名单模式）"
        except Exception as e:
            logger.error(f"cmd_tts_global_off failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    # ==================== 会话开关命令 ====================
    
    async def cmd_tts_on(self, event) -> str:
        """开启当前会话的 TTS。"""
        try:
            sid = self._sess_id(event)  # type: ignore
            if self.global_enable:  # type: ignore
                # 黑名单模式：从黑名单移除
                await self.config.remove_from_disabled_async(sid)  # type: ignore
                if sid in self.disabled_sessions:  # type: ignore
                    self.disabled_sessions.remove(sid)  # type: ignore
            else:
                # 白名单模式：加入白名单
                await self.config.add_to_enabled_async(sid)  # type: ignore
                if sid not in self.enabled_sessions:  # type: ignore
                    self.enabled_sessions.append(sid)  # type: ignore
            return "本会话TTS：开启"
        except Exception as e:
            logger.error(f"cmd_tts_on failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    async def cmd_tts_off(self, event) -> str:
        """关闭当前会话的 TTS。"""
        try:
            sid = self._sess_id(event)  # type: ignore
            if self.global_enable:  # type: ignore
                # 黑名单模式：加入黑名单
                await self.config.add_to_disabled_async(sid)  # type: ignore
                if sid not in self.disabled_sessions:  # type: ignore
                    self.disabled_sessions.append(sid)  # type: ignore
            else:
                # 白名单模式：从白名单移除
                await self.config.remove_from_enabled_async(sid)  # type: ignore
                if sid in self.enabled_sessions:  # type: ignore
                    self.enabled_sessions.remove(sid)  # type: ignore
            return "本会话TTS：关闭"
        except Exception as e:
            logger.error(f"cmd_tts_off failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    # ==================== 参数配置命令 ====================
    
    async def cmd_tts_prob(self, event, value: Optional[str] = None) -> str:
        """设置 TTS 触发概率。"""
        from ..core.constants import MIN_PROB, MAX_PROB
        
        try:
            if value is None:
                raise ValueError
            v = float(value)
            assert MIN_PROB <= v <= MAX_PROB
            self.prob = v  # type: ignore
            await self.config.set_prob_async(v)  # type: ignore
            return f"TTS概率已设为 {v}"
        except (ValueError, AssertionError):
            return "用法：tts_prob 0~1，如 0.35"
        except Exception as e:
            logger.error(f"cmd_tts_prob failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    async def cmd_tts_limit(self, event, value: Optional[str] = None) -> str:
        """设置 TTS 文本长度上限。"""
        try:
            if value is None:
                raise ValueError
            v = int(value)
            assert v >= 0
            self.text_limit = v  # type: ignore
            await self.config.set_text_limit_async(v)  # type: ignore
            return f"TTS字数上限已设为 {v}"
        except (ValueError, AssertionError):
            return "用法：tts_limit <非负整数>"
        except Exception as e:
            logger.error(f"cmd_tts_limit failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    async def cmd_tts_cooldown(self, event, value: Optional[str] = None) -> str:
        """设置 TTS 冷却时间。"""
        try:
            if value is None:
                raise ValueError
            v = int(value)
            assert v >= 0
            self.cooldown = v  # type: ignore
            await self.config.set_cooldown_async(v)  # type: ignore
            return f"TTS冷却时间已设为 {v}s"
        except (ValueError, AssertionError):
            return "用法：tts_cooldown <非负整数(秒)>"
        except Exception as e:
            logger.error(f"cmd_tts_cooldown failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    async def cmd_tts_gain(self, event, value: Optional[str] = None) -> str:
        """设置输出音量增益。"""
        from ..core.constants import MIN_GAIN, MAX_GAIN
        
        try:
            if value is None:
                raise ValueError
            v = float(value)
            assert MIN_GAIN <= v <= MAX_GAIN
            # 更新运行期
            try:
                self.tts.gain = v  # type: ignore
            except Exception:
                pass
            # 持久化
            await self.config.set_api_gain_async(v)  # type: ignore
            return f"TTS音量增益已设为 {v} dB"
        except (ValueError, AssertionError):
            return "用法：tts_gain <-10~10>，例：tts_gain 5"
        except Exception as e:
            logger.error(f"cmd_tts_gain failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    # ==================== 状态查询命令 ====================
    
    async def cmd_tts_status(self, event) -> str:
        """查询当前 TTS 状态。"""
        try:
            sid = self._sess_id(event)  # type: ignore
            mode = "黑名单(默认开)" if self.global_enable else "白名单(默认关)"  # type: ignore
            enabled = self._is_session_enabled(sid)  # type: ignore
            return (
                f"模式: {mode}\n"
                f"当前会话: {'启用' if enabled else '禁用'}\n"
                f"prob={self.prob}, limit={self.text_limit}, "  # type: ignore
                f"cooldown={self.cooldown}s, allow_mixed={self.allow_mixed}"  # type: ignore
            )
        except Exception as e:
            logger.error(f"cmd_tts_status failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    # ==================== 混合输出命令 ====================
    
    async def cmd_tts_mixed_on(self, event) -> str:
        """开启混合输出（文本+语音）。"""
        try:
            self.allow_mixed = True  # type: ignore
            await self.config.set_allow_mixed_async(True)  # type: ignore
            return "TTS混合输出：开启（文本+语音）"
        except Exception as e:
            logger.error(f"cmd_tts_mixed_on failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    async def cmd_tts_mixed_off(self, event) -> str:
        """关闭混合输出（仅纯文本时尝试合成）。"""
        try:
            self.allow_mixed = False  # type: ignore
            await self.config.set_allow_mixed_async(False)  # type: ignore
            return "TTS混合输出：关闭（仅纯文本时尝试合成）"
        except Exception as e:
            logger.error(f"cmd_tts_mixed_off failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    # ==================== 文字+语音会话级命令 ====================
    
    async def cmd_tts_text_voice_on(self, event) -> str:
        """当前会话开启文字+语音同时输出。"""
        try:
            sid = self._sess_id(event)  # type: ignore
            st = self._session_state.setdefault(sid, self._create_session_state())  # type: ignore
            st.text_voice_enabled = True
            return "当前会话：文字+语音同时输出 已开启"
        except Exception as e:
            logger.error(f"cmd_tts_text_voice_on failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    async def cmd_tts_text_voice_off(self, event) -> str:
        """当前会话关闭文字+语音同时输出。"""
        try:
            sid = self._sess_id(event)  # type: ignore
            st = self._session_state.setdefault(sid, self._create_session_state())  # type: ignore
            st.text_voice_enabled = False
            return "当前会话：文字+语音同时输出 已关闭（仅发送语音）"
        except Exception as e:
            logger.error(f"cmd_tts_text_voice_off failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    async def cmd_tts_text_voice_reset(self, event) -> str:
        """当前会话重置为跟随全局设置。"""
        try:
            sid = self._sess_id(event)  # type: ignore
            st = self._session_state.setdefault(sid, self._create_session_state())  # type: ignore
            st.text_voice_enabled = None
            return f"当前会话：文字+语音设置已重置，跟随全局（allow_mixed={self.allow_mixed}）"  # type: ignore
        except Exception as e:
            logger.error(f"cmd_tts_text_voice_reset failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    # ==================== 参考文献命令 ====================
    
    async def cmd_tts_check_refs(self, event) -> str:
        """检查参考文献配置。"""
        try:
            return (
                f"allow_mixed配置: {self.allow_mixed}\n"  # type: ignore
                f"配置文件中的allow_mixed: {self.config.get('allow_mixed', '未找到')}\n"  # type: ignore
                f"show_references配置: {self.show_references}\n"  # type: ignore
                f"配置文件中的show_references: {self.config.get('show_references', '未找到')}\n"  # type: ignore
                f"参考文献发送条件: {'满足' if self.show_references else '不满足 (需要开启 show_references)'}"  # type: ignore
            )
        except Exception as e:
            logger.error(f"cmd_tts_check_refs failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    async def cmd_tts_refs_on(self, event) -> str:
        """开启参考文献显示。"""
        try:
            self.show_references = True  # type: ignore
            await self.config.set_show_references_async(True)  # type: ignore
            return "参考文献显示：开启（包含代码或链接时会显示参考文献）"
        except Exception as e:
            logger.error(f"cmd_tts_refs_on failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    async def cmd_tts_refs_off(self, event) -> str:
        """关闭参考文献显示。"""
        try:
            self.show_references = False  # type: ignore
            await self.config.set_show_references_async(False)  # type: ignore
            return "参考文献显示：关闭（包含代码或链接时不会显示参考文献）"
        except Exception as e:
            logger.error(f"cmd_tts_refs_off failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    # ==================== 测试和调试命令 ====================
    
    async def cmd_tts_test(self, event, text: Optional[str] = None):
        """
        测试 TTS 功能并诊断问题。
        
        Returns:
            生成器，产出多条消息
        """
        from ..core.constants import TEMP_DIR, EMOTIONS, DEFAULT_TEST_TEXT
        from ..utils.audio import ensure_dir, validate_audio_file
        
        if not text:
            text = DEFAULT_TEST_TEXT
        
        sid = self._sess_id(event)  # type: ignore
        if not self._is_session_enabled(sid):  # type: ignore
            yield "本会话TTS未启用，请使用 tts_on 启用"
            return
        
        try:
            # 选择默认情绪和音色
            emotion = "neutral"
            vkey, voice = self._pick_voice_for_emotion(emotion)  # type: ignore
            if not voice:
                yield f"错误：未配置音色映射，请先配置 voice_map.{emotion}"
                return
            
            # 创建输出目录
            out_dir = TEMP_DIR / sid
            ensure_dir(out_dir)
            
            # 生成音频
            yield f"正在生成测试音频：\"{text}\"..."
            
            start_time = time.time()
            audio_path = await self.tts.synth(text, voice, out_dir, speed=None)  # type: ignore
            generation_time = time.time() - start_time
            
            if not audio_path:
                yield "❌ TTS API调用失败"
                return
            
            # 验证文件
            if not await validate_audio_file(audio_path):
                yield f"❌ 生成的音频文件无效: {audio_path}"
                return
            
            # 路径规范化测试
            normalized_path = self._normalize_audio_path(audio_path)  # type: ignore
            
            # 尝试创建 Record 对象
            try:
                from ..core.compat import import_message_components
                comps = import_message_components()
                Record = comps[0]
                record = Record(file=normalized_path)
                record_status = "✅ 成功"
            except Exception as e:
                record_status = f"❌ 失败: {e}"
            
            # 报告结果
            file_size = audio_path.stat().st_size
            result_msg = f"""🎵 TTS测试结果：
✅ 音频生成成功
📁 文件路径: {audio_path.name}
📊 文件大小: {file_size} 字节
⏱️ 生成耗时: {generation_time:.2f}秒
🎯 使用音色: {vkey} ({voice[:30]}...)
📝 Record对象: {record_status}
🔧 规范化路径: {normalized_path == str(audio_path)}"""
            
            yield result_msg
            
            # 尝试发送音频（需要由调用方处理）
            yield ("__AUDIO__", str(audio_path))
            
        except Exception as e:
            logger.error(f"cmd_tts_test failed: {e}", exc_info=True)
            yield f"❌ TTS测试失败: {e}"
    
    async def cmd_tts_debug(self, event) -> str:
        """显示 TTS 调试信息。"""
        try:
            import platform
            import os
            from ..core.constants import TEMP_DIR, EMOTIONS
            from ..core.session import SessionState
            
            sid = self._sess_id(event)  # type: ignore
            st = self._session_state.get(sid, SessionState())  # type: ignore
            
            debug_info = f"""🔧 TTS调试信息：
🖥️ 系统: {platform.system()} {platform.release()}
📂 Python路径: {os.getcwd()}
🆔 会话ID: {sid}
⚡ 会话状态: {'✅ 启用' if self._is_session_enabled(sid) else '❌ 禁用'}
🎛️ 全局开关: {'✅ 开启' if self.global_enable else '❌ 关闭'}
🎲 触发概率: {self.prob}
📏 文字限制: {self.text_limit}
⏰ 冷却时间: {self.cooldown}s
🔄 混合内容: {'✅ 允许' if self.allow_mixed else '❌ 禁止'}
🎵 API模型: {self.tts.model}
🎚️ 音量增益: {self.tts.gain}dB
📁 临时目录: {TEMP_DIR}

📊 会话统计:
🕐 最后TTS时间: {time.strftime('%H:%M:%S', time.localtime(st.last_tts_time)) if st.last_tts_time else '无'}
📝 最后TTS内容: {st.last_tts_content[:30] + '...' if st.last_tts_content and len(st.last_tts_content) > 30 else st.last_tts_content or '无'}
😊 待用情绪: {st.pending_emotion or '无'}

🎭 音色配置:"""  # type: ignore
            
            for emotion in EMOTIONS:
                vkey, voice = self._pick_voice_for_emotion(emotion)  # type: ignore
                speed = self.speed_map.get(emotion) if isinstance(self.speed_map, dict) else None  # type: ignore
                debug_info += f"\n{emotion}: {vkey if voice else '❌ 未配置'}"
                if speed:
                    debug_info += f" (语速: {speed})"
            
            return debug_info
        except Exception as e:
            logger.error(f"cmd_tts_debug failed: {e}", exc_info=True)
            return f"错误: {e}"
    
    # ==================== 辅助方法 ====================
    
    def _create_session_state(self):
        """创建新的会话状态（由主类实现）。"""
        from ..core.session import SessionState
        return SessionState()