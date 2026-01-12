# 🎭 AstrBot TTS 情绪路由插件

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/muyouzhi6/astrbot_plugin_tts_emotion_router)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

> 智能情绪识别的 TTS 插件，根据对话情绪自动切换音色与语速，让你的 AstrBot 更有感情！

## ✨ 核心特性

### 🧠 智能情绪识别
- **隐藏标记解析**：支持 `[EMO:happy]`、`【EMO：开心】`、`&happy&` 等多种情绪标签格式
- **启发式分类**：无标记时自动分析文本情绪（开心、难过、愤怒、中性）
- **上下文感知**：结合对话历史进行情绪判断

### 🎵 音色语速路由
- **情绪音色映射**：不同情绪自动使用不同的语音音色
- **动态语速调节**：开心加速 1.2x，难过减速 0.85x，完美还原情感
- **硅基流动 API**：支持 CosyVoice 等高质量 TTS 模型

### 🛡️ 智能文本处理
- **代码块过滤**：自动识别并跳过 ```代码``` 和 `行内代码`
- **表情符号处理**：智能过滤 emoji 和 QQ 表情，避免朗读 `[微笑]` `😀`
- **链接检测**：自动跳过 URL 和文件路径
- **文本优化**：防止 TTS 吞字，确保完整朗读
- **参考文献控制**：可选择是否显示提取的代码和链接的参考文献

### ⚡ 高级控制
- **会话级开关**：每个对话独立控制 TTS 开关
- **概率门控**：可设置触发概率，避免过度朗读
- **长度限制**：超长文本自动跳过，专注短句对话
- **冷却机制**：防止频繁触发，优化体验
- **系统指令识别**：自动识别系统指令（如 /help、/provider 等），直接文本输出，不进行TTS转换

## 🚀 快速开始（60 秒上手）

### ✅ 你实际只需要做这些
1. 在 AstrBot 后台【插件市场】搜索并安装 / 启用 “astrbot_plugin_tts_emotion_router”。  
2. 安装 ffmpeg（系统仅需能调用 ffmpeg 命令）。  
3. 拥有硅基流动（或兼容 OpenAI TTS 接口）的 API Key。  
4. （可选）使用提供的音色上传工具压缩包导入自定义音色
   方法（推荐）：[voice.gbkgov.cn](https://voice.gbkgov.cn/)【特别鸣谢Chris的硅基音色一键上传网站】
  
6. 在插件配置里填写 api 信息与 voice_map / speed_map。
   
6.**配置检验**
   发送/tts_test 哈喽哈喽，出现以下回复就说明插件配置没问题（第一条正常就行，忽略第二条报错）
   ![b954f3db3b2c9cabb4814b920b931e69_720](https://github.com/user-attachments/assets/cabc39be-e80d-4e1d-8792-7434606a8031)
完成！

### 📋 运行所需
- AstrBot v3.5+ 已运行
- ffmpeg（必需）
- 硅基流动 API Key（或其它兼容语音合成服务）

### 📦 ffmpeg 安装示例
```bash
# Linux (Debian/Ubuntu)
sudo apt install -y ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey 管理员 PowerShell)
choco install ffmpeg -y
```
安装后确认：`ffmpeg -version` 输出版本即可。

### 🎙 自定义音色（可选）
1. 下载 README 下方链接的 “硅基音色一键上传” 压缩包。
2. 解压并按提示运行（通常是一个脚本或可执行程序）。
3. 录入或选择音色素材，上传生成对应的自定义音色标识。
4. 在插件配置 `voice_map` 中填入生成的音色名（格式示例：`FunAudioLLM/CosyVoice2-0.5B:anna` 或你的自定义 ID）。

### 🧪 验证与调试
```bash
tts_status          # 查看当前插件总体状态
tts_debug "你好，今天天气不错！"  # 观察情绪识别 + 过滤效果
tts_refs_on / tts_refs_off       # 开启/关闭参考文献显示
```

### 🛠 离线/手动安装（很少需要）
仅在无法访问插件市场时：将发布的 zip/源码目录解压到 `data/plugins/astrbot_plugin_tts_emotion_router`，重启 AstrBot；其余步骤相同。

> 旧文档中的 `git clone` / `uv sync` 步骤对普通使用者已不再必需，故已简化。

### 推荐可与 STT 插件配合实现与 Bot 全语音交流
https://github.com/NickCharlie/Astrbot-Voice-To-Text-Plugin
### ⚙️ 基础配置

<details>
<summary><b>🔧 API 配置（必填）</b></summary>

| 配置项 | 示例值 | 说明 |
|--------|--------|------|
| `api.url` | `https://api.siliconflow.cn/v1` | API 服务地址 |
| `api.key` | `sk-xxx` | 你的 API Key |
| `api.model` | `FunAudioLLM/CosyVoice2-0.5B` | TTS 模型名称 |
| `api.format` | `mp3` | 音频格式（推荐 mp3） |
| `api.speed` | `1.0` | 默认语速 |

</details>

<details>
<summary><b>🎭 情绪路由配置</b></summary>

```yaml
# 音色映射（必须至少配置 neutral）
voice_map:
  neutral: "FunAudioLLM/CosyVoice2-0.5B:anna"    # 中性音色
  happy: "FunAudioLLM/CosyVoice2-0.5B:cheerful"  # 开心音色
  sad: "FunAudioLLM/CosyVoice2-0.5B:gentle"      # 难过音色
  angry: "FunAudioLLM/CosyVoice2-0.5B:serious"   # 愤怒音色
  # 自定义音色从"speech"开始填写
  # neutral: "speech:sad-zhizhi-voice:icwcmuszkb:vdpjnvpfqbqbsywmbyly"

# 语速映射（自己按喜欢设置）
speed_map:
  neutral: 1.0    # 正常语速
  happy: 1.2      # 开心加速
  sad: 0.85       # 难过减速
  angry: 1.1      # 愤怒略快
```

</details>

<details>
<summary><b>🏷️ 情绪标签配置</b></summary>

```yaml
emotion:
  marker:
    enable: true        # 启用隐藏标记解析
    tag: "EMO"          # 标签名称，对应 [EMO:happy]
    prompt_hint: |      # 复制到系统提示中
      请在每次回复末尾追加形如 [EMO:<happy|sad|angry|neutral>] 的隐藏标记。
      该标记仅供系统解析，不会展示给用户。
```

</details>

## 🎯 使用指南

### 💡 推荐配置流程
<img width="580" height="1368" alt="PixPin_2025-08-25_17-00-01" src="https://github.com/user-attachments/assets/6cd57fb9-9b39-4dae-80e4-c9bd0c3400de" />

1. **配置系统提示**
   在你的 AI 人格设定中添加：
   ```
   请在每次回复末尾追加形如 [EMO:<happy|sad|angry|neutral>] 的隐藏标记。
   该标记仅供系统解析，不会展示给用户。开心时用 happy，难过时用 sad，
   愤怒时用 angry，其他情况用 neutral。
   ```

2. **测试情绪识别**
   ```
   用户：今天天气真不错！[EMO:happy]
   机器人：是的，阳光明媚的日子总是让人心情愉悦呢！[EMO:happy]
   ```

3. **调试和优化**
   ```bash
   tts_test_problematic  # 测试问题文本处理
   tts_debug 测试文本   # 查看处理过程
   tts_status           # 查看当前状态
   ```

   
### 🎮 会话控制命令

| 命令 | 功能 | 示例 |
|------|------|------|
| `tts_on` | 当前会话启用 | 开启语音回复 |
| `tts_off` | 当前会话禁用 | 关闭语音回复 |
| `tts_global_on` | 全局启用 | 所有会话默认开启 |
| `tts_global_off` | 全局禁用 | 所有会话默认关闭 |
| `tts_prob 0.8` | 设置触发概率 | 80% 概率触发 |
| `tts_limit 100` | 设置长度限制 | 超过100字跳过 |
| `tts_cooldown 30` | 设置冷却时间 | 30秒内不重复 |
| `tts_status` | 查看状态 | 显示当前配置 |
| `tts_refs_on/off` | 参考文献开关 | 控制是否显示代码和链接的参考文献 |

### 🏷️ 支持的情绪标签格式

插件支持多种情绪标签格式，会自动识别并移除：

```
标准格式：[EMO:happy] [EMO:sad] [EMO:angry] [EMO:neutral]
中文格式：【EMO：开心】【EMO：难过】【EMO：愤怒】【EMO：中性】
简短格式：happy: sad: angry: neutral:
符号格式：&happy& &sad& &angry& &neutral&
情绪格式：【情绪：开心】【情绪：难过】
```

### 🛡️ 智能文本过滤示例

```python
# 原始文本
"看看这个代码 `print('hello')` 很简单吧！😊 [微笑]"

# 处理后文本（用于TTS）
"看看这个代码很简单吧！"

# 被过滤的内容
- 行内代码：`print('hello')`
- Emoji：😊
- QQ表情：[微笑]
```

## 🔧 故障排除

### ❌ 常见问题

<details>
<summary><b>Q: 没有语音输出，只有文字</b></summary>

**可能原因：**
1. API 配置错误
2. 网络连接问题
3. ffmpeg 未安装
4. 音色配置不存在

**解决步骤：**
```bash
# 1. 检查 API 配置
tts_debug 测试文本

# 2. 检查网络
curl -I https://api.siliconflow.cn/v1/audio/speech

# 3. 检查 ffmpeg
ffmpeg -version

# 4. 检查日志
# 查看 AstrBot 控制台输出的错误信息
```

</details>

<details>
<summary><b>Q: 情绪不切换，总是同一个音色</b></summary>

**解决方案：**
1. **启用标记解析**：确保 `emotion.marker.enable = true`
2. **配置音色映射**：检查 `voice_map` 中各情绪音色是否配置
3. **系统提示**：在 AI 设定中添加情绪标记提示
4. **测试启发式**：发送明显情绪的文本（如"太棒了！"）

</details>

<details>
<summary><b>Q: TTS 朗读了代码块和表情符号</b></summary>

这已经在最新版本中修复！插件会自动过滤：
- Markdown 代码块：```代码```
- 行内代码：`代码`
- Emoji 表情：😊 🎉 等
- QQ 表情：[微笑] [大笑] 等
- 情绪标签：所有格式的情绪标记

</details>

<details>
<summary><b>Q: 概率、长度、冷却不生效</b></summary>

**检查配置：**
```bash
tts_status  # 查看当前设置

# 重新设置
tts_prob 1.0      # 100% 触发
tts_limit 999     # 取消长度限制  
tts_cooldown 0    # 取消冷却
```

</details>

### 🔍 调试工具

```bash
# 文本处理调试
tts_debug "你好！😊 [EMO:happy]"

# 问题文本测试
tts_test_problematic

# 查看会话状态
tts_status

# 检查插件日志
# 在 AstrBot 控制台查看详细输出
```

## 🎨 高级用法

### 🎭 自定义音色上传

使用音色上传工具：[硅基音色一键上传.zip](https://github.com/user-attachments/files/22064355/default.zip)


**需要准备：**
- 5MB 以下的 10 秒左右清晰人声素材
- 对应的文本内容
- 硅基流动 API Key（建议独立申请）

### 🔄 与其他插件配合

**STT 语音识别插件：**
```
配合 https://github.com/NickCharlie/Astrbot-Voice-To-Text-Plugin
实现完整的语音对话体验：语音输入 → 文字回复 → 情绪 TTS
```

### ⚡ 性能优化

```yaml
# 高频对话场景
prob: 0.6           # 降低触发概率
text_limit: 50      # 限制长文本
cooldown: 15        # 增加冷却时间

# 低延迟场景  
api.format: "mp3"   # 使用 MP3 格式
max_retries: 1      # 减少重试次数
timeout: 15         # 缩短超时时间
```

## 🔬 技术原理

### 🧠 情绪分类算法

1. **隐藏标记优先**：检测并解析 `[EMO:xxx]` 等格式标记
2. **启发式分析**：基于关键词、标点符号、上下文推断
3. **规则引擎**：
   - 积极词汇 → happy（开心、太棒了、哈哈）
   - 消极词汇 → sad（难过、失望、唉）  
   - 愤怒词汇 → angry（生气、愤怒、气死）
   - 信息性内容 → neutral（链接、代码等）

### 🎵 音色路由机制

```python
情绪识别 → 查找音色映射 → 应用语速调节 → TTS 合成 → 音频输出
    ↓           ↓           ↓          ↓        ↓
  happy    → cheerful  →   1.2x   →   API   → xxx.mp3
```

### 🛡️ 文本处理流程

```
原始文本 → 过滤代码块 → 过滤表情 → 清理标签 → 添加结尾 → TTS合成
```

## 📈 版本历史
- **v1.0.0** (当前)
   - 🚀 版本更新：1.全面重构优化性能。
                 2.新增指令 tts_text_voice_on、tts_text_voice_off，支持单独开启当前对话文字语音同时输出（为了适配桌面助手）
- **v0.5.0** 
   - 🚀 版本更新：功能增强与稳定性优化。

- **v0.4.4**
   - ⚡ 优化情绪分类：信息类文本（包含链接/代码等）直接判为 neutral，避免误判为 happy/sad。
   - 🔧 调整阈值：微弱情绪分值（<=0.5）归为 neutral，降低感叹号的情绪权重。
   - 🧩 优化上下文感知：过滤无效上下文，降低上下文对当前情绪的干扰。

- **v0.4.3**
   - 🧩 代码/链接提取大幅优化：减少普通词、文件名、伪域名误判；支持去重、域名白名单、伪方法链过滤。
   - 🔍 行内代码智能判定：仅保留真正变量/函数/表达式；忽略模型名、版本号、纯扩展名及简单文件名。
   - 🔗 链接处理：去除重复、半截文件名与方法链不再当作链接；支持合法 TLD 过滤；保留真实 URL。
   - 📚 参考文献可配置：新增 `references.preview_limit` 与 `references.max_total_chars`；默认不截断完整显示。
   - 🧪 新增扩展测试用例覆盖 tavily、TAVILY_API_KEY、hello.py、plugin.html、伪域名等场景。
   - 🛠️ 结构清理：提取逻辑模块化，便于后续扩展调试模式。
   - ⚙️ 兼容旧行为：默认配置保持与 0.3.x 一致（显示参考文献，不截断代码）。

- **v0.3.2**
   - 📚 新增参考文献显示开关功能，用户可选择是否显示代码和链接的参考文献
   - 🎮 新增 `tts_refs_on/off` 命令，方便动态控制参考文献显示
   - 🛠️ 优化 `tts_check_refs` 命令，显示更详细的配置信息

- **v0.3.1**
- 🎯 新增系统指令识别功能，自动跳过 help、provider 等系统命令的TTS转换
- 🛡️ 优化消息类型判断逻辑，更准确地区分LLM响应和系统响应
- 📝 完善日志输出，便于调试和问题排查

- **v0.3.0**
- ✨ 新增代码和链接智能提取功能
- 🎯 优化输出体验，避免内容重复
- 📖 移除多余的标题文字，展示更简洁
- 🛠️ 修复 allow_mixed 配置的相关问题


- **v0.2.1** 
  - 🆕 新增代码块和表情符号过滤
  - 🆕 支持 `&emotion&` 标签格式
  - 🔧 修复 TTS 吞字问题
  - 🔧 修复事件传播中断问题
  - ⚡ 优化文本处理性能

- **v0.1.0**
  - 🎉 首个发布版本
  - ✅ 基础情绪识别和音色路由
  - ✅ 硅基流动 API 集成

## 🤝 参与贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交改动：`git commit -m '添加某个特性'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

## 📄 开源协议

本项目基于 MIT 协议开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 👨‍💻 作者信息

- **作者**：木有知
- **仓库**：https://github.com/muyouzhi6/astrbot_plugin_tts_emotion_router
- **版本**：0.5.0

---

<div align="center">
  
**🌟 如果这个插件对你有帮助，请给个 Star 支持一下！**

*让每一句话都充满感情！*

</div>
















