# 🎭 AstrBot TTS 情绪路由插件

[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/muyouzhi6/astrbot_plugin_tts_emotion_router)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

面向中文场景的 TTS 插件：支持情绪路由、会话黑白名单策略、分段语音、概率语音，以及按需触发语音输出。

## ✨ 功能概览

- 支持双 TTS 服务商：`siliconflow` / `minimax`
- 情绪路由：按 `happy/sad/angry/neutral` 映射不同音色和语速
- 四类可独立配置策略（均支持 UMO 黑白名单）
  - 自动语音输出 `voice_output`
  - 文字+语音同时输出 `text_voice_output`
  - 分段语音输出 `segmented_output`
  - 概率语音输出 `probability_output`
- 可关闭所有强制自动语音，改为按需触发
  - 命令：`tts_say`
  - LLM 工具：`tts_speak`

## 🚀 快速开始

1. 在 AstrBot 插件市场安装并启用本插件。
2. 安装 `ffmpeg`（系统可直接调用即可）。
3. 在插件配置面板填写 TTS 参数（推荐先用 SiliconFlow 跑通）。
4. 在群聊或私聊发送：
   - `/sid` 获取当前会话 UMO
   - `tts_status` 查看当前插件状态
   - `tts_say` 测试语音输出（不填文本会用默认测试语句）

配置成功示例：

![b954f3db3b2c9cabb4814b920b931e69_720](https://github.com/user-attachments/assets/cabc39be-e80d-4e1d-8792-7434606a8031)

## ⚙️ 核心配置说明

### 1) UMO 与黑白名单

- 在聊天中发送 `/sid` 获取 UMO。
- 所有黑白名单字段都填写 UMO。
- 面板中的 `mode`：
  - `blacklist`：默认开启，命中黑名单关闭
  - `whitelist`：默认关闭，仅白名单开启

### 2) TTS 引擎

- `tts_engine.provider` 可选：
  - `siliconflow`
  - `minimax`

### 3) MiniMax（已支持）

接口使用 `https://api.minimaxi.com/v1/t2a_v2`，支持：

- `model`
- `voice_id`
- `speed`
- `vol`
- `pitch`
- `emotion`
- `audio_format`
- `sample_rate`
- `bitrate`
- `channel`
- `subtitle_enable`
- `pronunciation_dict`（可选）

### 4) 情绪路由面板

- `emotion_route.enable = true` 时显示子面板：
  - `voice_map`
  - `speed_map`
  - `marker`
  - `keywords`
- 关闭后不显示映射配置，界面更简洁。

### 5) 分段语音参数

- `segmented_tts.enable`
- `interval_mode = fixed/adaptive`
- `fixed_interval`
- `adaptive_buffer`
- `max_segments`
- `min_segment_chars`
- `split_pattern`
- `min_segment_length`

## 🧩 当前可用命令

| 命令 | 说明 |
|------|------|
| `tts_on` | 开启当前会话语音输出 |
| `tts_off` | 关闭当前会话语音输出 |
| `tts_all_on` | 开启全局自动语音输出 |
| `tts_all_off` | 关闭全局自动语音输出（保留按需语音） |
| `tts_status` | 查看当前状态（含 UMO、策略命中情况） |
| `tts_say [文本]` | 手动发一条语音（不填文本使用默认测试文本） |

LLM 工具（函数调用）：

- `tts_speak(text: str)`：在需要时由 Bot 主动输出语音

## 🎙 音色克隆与上传（保留）

你原 README 中的音色上传能力已保留，建议顺序如下：

1. 准备清晰人声音频（建议 10 秒左右，尽量纯净）。
2. 上传并生成可用音色 ID。
3. 把音色 ID 填入 `voice_map`（至少配置 `neutral`）。
4. 用 `tts_say` 验证。

推荐入口（保留）：

- 一键上传站点：<https://voice.gbkgov.cn/>（特别鸣谢 Chris）
- 上传工具压缩包：  
  [硅基音色一键上传.zip](https://github.com/user-attachments/files/22064355/default.zip)

`voice_map` 示例：

```yaml
emotion_route:
  enable: true
  voice_map:
    neutral: "FunAudioLLM/CosyVoice2-0.5B:anna"
    happy: "FunAudioLLM/CosyVoice2-0.5B:cheerful"
    sad: "FunAudioLLM/CosyVoice2-0.5B:gentle"
    angry: "FunAudioLLM/CosyVoice2-0.5B:serious"
  speed_map:
    neutral: 1.0
    happy: 1.2
    sad: 0.85
    angry: 1.1
```

## 💬 群聊展示与交流（保留）

推荐配置流程展示：

<img width="580" height="1368" alt="PixPin_2025-08-25_17-00-01" src="https://github.com/user-attachments/assets/6cd57fb9-9b39-4dae-80e4-c9bd0c3400de" />

插件开发交流群（保留）：

- QQ 群：`215532038`

<img width="1284" height="2289" alt="qrcode_1767584668806" src="https://github.com/user-attachments/assets/113ccf60-044a-47f3-ac8f-432ae05f89ee" />

## 🔄 与 STT 插件配合

可配合以下 STT 插件实现完整语音交互：

<https://github.com/NickCharlie/Astrbot-Voice-To-Text-Plugin>

流程示例：语音输入 -> 文本理解 -> 情绪路由 TTS 输出

## 🛠 常见排查

### 没有语音输出

1. 先看 `tts_status` 是否命中黑白名单策略。
2. 检查服务商配置是否完整（特别是 `key`、`voice_map.neutral`）。
3. 检查 `ffmpeg -version` 是否可执行。
4. 检查网络与 API 可用性。

### 已关闭自动语音但仍想让 Bot 说话

- 用 `tts_all_off` 关闭自动语音
- 在需要时使用 `tts_say`
- 或让模型函数调用 `tts_speak`

### 情绪没有切换

1. 确保 `emotion_route.enable = true`。
2. 确保 `voice_map` 至少配置了 `neutral`，建议四种情绪都配。
3. 检查 `speed_map` 是否存在异常值。

## 📌 项目信息

- 作者：木有知（muyouzhi6）
- 仓库：<https://github.com/muyouzhi6/astrbot_plugin_tts_emotion_router>
- 协议：MIT

---

如果这个插件对你有帮助，欢迎点个 Star。
