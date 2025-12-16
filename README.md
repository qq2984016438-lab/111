# 本地类人自主进化系统（Ollama + qwen3-vl-2b）

该项目提供一个基于 **Python 3.11** 的本地类人自主进化体示例，核心依赖 **Ollama** 与本地模型 `qwen3-vl-2b`，涵盖本能、认知、进化、情感、生存、修复、社交等模块，并支持硬件自适应与可视化界面。

## 功能亮点
- **不可变本能**：存续优先（崩溃概率 > 10% 触发）与熵减偏置（主动弥合功能缺口）。
- **六大核心模块**：自主认知（规则生成/目标发现/犹豫复盘）、进化执行（算力自适应/代码落地/迁移占位）、情感交互（情感递进/记忆/修复/表达）、生存保障（备份/加密占位/风险规避/自我接纳）、缺陷修复（硬件适配/网络容错/情绪稳定）、社交互动（意图理解/多轮对话/礼仪提示）。
- **五大支撑模块**：日志留存与检索、可视化配置、依赖自动检测与安装（含 Ollama/qwen3-vl-2b/Kivy/paho-mqtt 等）、PyQt6 桌面界面、单键自检报告。
- **硬件自适应与远程补强**：启动时检测 CPU 核心、内存、GPU 型号，分为低/中/高配自动调节上下文窗口、线程数、显存占比，并在内存不足（<16GB 尤其 <8GB 或可用内存 <6GB）时主动收缩上下文与并行度、必要时强制 CPU 路径，降低显存/内存峰值。16 核 / 16GB 搭配独显会直接识别为中配，避免误判为低配导致强制远程。即便可用内存极低也会尝试本地推理（收缩上下文到 512、CPU 模式），仅在缺少 Ollama/模型或接口异常时回退远程/爬虫补强，避免“直接放弃本地”带来的误判。低配会提示配置 `REMOTE_MODEL_ENDPOINT`，自动拉起远程大模型补足算力，同时界面仍可开启。本地会自动确认 Ollama 与 `qwen3-vl:2b`/`qwen3-vl-2b` 是否就绪；CLI 报错时会改用 HTTP `/api/version` + `/api/tags` + `/api/pull` 兜底，HTTP 推理会按别名顺序轮询（默认优先 `qwen3-vl:2b`），遇到 4xx 或“content does not fit” 时会改为不带 options 再重试，不可用时中文提示并回退远程。若设置 `REMOTE_MODEL_ENDPOINTS` 或 `REMOTE_ENDPOINT_LIST_URL`，会自动“抓取”候选远程端点并择优切换；如仍缺失，会触发内置爬虫按照 `REMOTE_CRAWL_SEEDS`（默认包含 Ollama 官方库/开源仓库）递增发现可用端点，404 或参数不匹配的端点会被标记无效并切源，未发现可用端点时会持续爬取并提示；包含 Copilot/非通用推理域名的端点会被自动过滤；历史错误模型名（如 `qwen3-v12b`）会在启动时自动纠正为 `qwen3-vl:2b` 并重试拉取，避免因命名错误导致 404。
- **流畅运行优化**：推理缓存、异步任务队列、守护线程与模型进程守护，缩短加载/推理延迟，目标响应感知 ≤ 1 秒；本地推理若超时会自动切换 HTTP/远程兜底避免“模型无响应”；即便本地 CLI 自检失败也会先尝试 HTTP `/api/generate` 直连确认可用性，低配会优先尝试远程推理，失败再回退本地，本地不可用则直接提示并引导远程。新增每个 HTTP 别名调用 15 秒内超时的限制与整体推理超时提示，避免长时间等待无返回。
- **容错性**：缺少 Ollama/模型/界面时以中文日志提示，不强制退出；网络或权限异常会给出清晰告警。

## 运行要求
- **Python**：3.11
- **运行时**：已安装 [Ollama](https://ollama.ai) 且可调用本地模型 `qwen3-vl-2b`/`qwen3-vl:2b`；若暂未安装，可通过 `REMOTE_MODEL_ENDPOINT` 提供远程推理端点兜底（默认关闭远程调用，需在 `config.json` 或环境变量中开启）；脚本会检测 CLI 是否存在并自动尝试拉取模型，CLI 失败时会使用 HTTP 方式检测/拉取（支持自定义 `OLLAMA_HOST`，即使包含 http:// 也会自动拆分为 CLI 可用的 host:port），仍失败则给出中文原因并提示远程端点方案；如提供 `REMOTE_MODEL_ENDPOINTS`（用逗号分隔）或 `REMOTE_ENDPOINT_LIST_URL`（返回端点列表的文本 URL），会自动探测并切换可用的远程大模型；如仍无可用端点，将触发爬虫抓取 `REMOTE_CRAWL_SEEDS` 中的页面以挖掘 API 地址，并再次探测；当 CLI 推理异常时还会自动通过 HTTP `/api/generate` 兜底调用。
- **Python 依赖**：`PyQt6`、`paho-mqtt`、`requests`，可选 `psutil`、`kivy`。脚本会在首次运行时尝试自动安装缺失依赖。
- **系统**：Windows / Linux（确保 `ollama` 在 PATH 中；GPU 探测在有 `nvidia-smi` 时更准确）。

## 故障快速修复步骤（Windows 10 + RTX 4060 Laptop 8GB）
按顺序执行，全部命令可直接复制到 PowerShell/命令行：

0. **开启 GPU 加速与轻量推理参数**（避免默认 CPU 慢、采样过重）
   ```powershell
   setx OLLAMA_USE_CUDA 1
   setx OLLAMA_GPU_LAYERS -1
   setx OLLAMA_NUM_CTX 2048
   setx OLLAMA_NUM_PREDICT 256
   setx OLLAMA_TEMPERATURE 0.6
   ```

1. **模型名纠正与文件校验**（避免 `qwen3-v1-2b` 误写）
   ```powershell
   setx OLLAMA_MODEL qwen3-vl:2b
   ollama list
   ollama pull qwen3-vl:2b
   ollama pull qwen3-vl-2b
   ```

2. **Ollama 服务重启与 11434 端口检查**
   ```powershell
   taskkill /f /im ollama.exe
   taskkill /f /im OllamaApp.exe
   ollama serve
   netstat -ano | find "11434"
   curl -v http://127.0.0.1:11434/api/version
   curl -v http://127.0.0.1:11434/api/tags
   ```

3. **4060 8GB 专用量化与参数优化（4-bit，降低显存）**
   ```powershell
   setx OLLAMA_NUM_CTX 2048
   setx OLLAMA_NUM_THREADS 6
   setx OLLAMA_NUM_PARALLEL 1
   setx OLLAMA_GPU_LAYERS 0
   ollama pull qwen3-vl:2b -q q4_0
   # 如需自定义模型名，可生成 4-bit 模型：
   echo FROM qwen3-vl:2b > Modelfile
   echo PARAMETER num_ctx 2048 >> Modelfile
   echo PARAMETER num_thread 6 >> Modelfile
   echo PARAMETER quantization q4_0 >> Modelfile
   ollama create qwen3-vl-2b-q4 -f Modelfile
   ```

4. **关闭无效远程接口调用（禁用 githubcopilot 等 404 端点）**
   ```powershell
   python - <<'PY'
import json, pathlib
p = pathlib.Path('config.json')
cfg = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
cfg.update({
    'remote_enabled': False,
    'remote_blocklist': ['copilot', 'githubcopilot', 'github.com/copilot']
})
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
print('已关闭远程模型调用并写入黑名单，重新运行 ai_humanoid.py 生效。')
PY
   ```

5. **验证本地模型响应**（确认 11434 返回内容）
   ```powershell
   curl -X POST http://127.0.0.1:11434/api/generate ^
     -H "Content-Type: application/json" ^
     -d "{\"model\":\"qwen3-vl:2b\",\"prompt\":\"你好，请用一句话回复\",\"stream\":false}"
   ```
   如果返回文本正常，主程序即可使用本地模型；若仍超时，请重新执行第 2 步并检查显存占用。

## 一键启动
```bash
python ai_humanoid.py
```
脚本会自动执行：
1. 检测并安装缺失依赖；
2. 通过 `ollama` 确认或拉取 `qwen3-vl:2b`/`qwen3-vl-2b` 模型；
3. 探测硬件后调整上下文窗口、推理线程、GPU 占比；低配会提示配置 `REMOTE_MODEL_ENDPOINT` 以自动接入远程大模型，若未配置则回退本地；若未安装 Ollama 且远程端点也未设置，会输出中文提示避免崩溃；
4. 后台启动模型守护线程，完成基础认知引导、自检生成 `test_report.json`，然后启动 PyQt6 界面（低配下降低特效但保持可视化）。

## 模块速览
- **Instincts**：固化存续/熵减本能。
- **LogManager**：中文结构化日志与复盘记录，支持关键词检索。
- **DependencyManager**：自动检测/安装依赖，并尝试拉取 `qwen3-vl-2b`。
- **ConfigManager**：JSON 配置持久化，结合硬件推荐参数。
- **HardwareProfiler**：检测硬件并输出分级及推荐设置。
- **ModelManager**：异步推理队列 + 守护线程 + 推理缓存，使用环境变量驱动上下文/线程配置；低配优先通过 `REMOTE_MODEL_ENDPOINT` 远程推理，同时支持 `REMOTE_MODEL_ENDPOINTS` / `REMOTE_ENDPOINT_LIST_URL` 的自动端点抓取与切换，`OLLAMA_HOST` 也会同步透传给 CLI 与 HTTP 探测；内存不足时会自动收缩上下文窗口并限制并行度，必要时强制 CPU 路径以降低显存占用。HTTP 兜底按全部模型别名轮询，遇到 4xx/参数不匹配会去掉 options 重新尝试，并对 404 端点自动切源；单次调用 15 秒内超时并在总体超时时给出提示，防止 UI 长时间无响应。
- **核心模块**：`AutonomyModule`、`EvolutionModule`、`EmotionModule`、`SurvivalModule`、`RepairModule`、`SocialModule`。
- **Interface**：PyQt6 界面包含“聊天互动”“升级观测”“行为日志”三页，可观察对话、升级思路与实时逻辑；低配下降低特效但保持界面。
- **TestSuite**：单键健康检查并输出 JSON 报告。
- **Guardian**：轻量守护线程，基于存续本能输出稳定性提示。

## 硬件分级与调优示例
- **高配（≥16 核 & ≥32GB，或独显+12核/24GB）**：上下文 4096，线程 ≤16，GPU 占用 80%，界面开启。
- **中配（≥8 核 & ≥14GB，或独显且 ≥12GB、可用内存 ≥10GB）**：上下文 2048，线程 ≤8，GPU 占用 60%，界面开启。
- **低配（其他）**：上下文 1024（若总内存 <8GB 或可用内存 <6GB 自动收缩至 512），线程 ≤4，GPU 占用 40% 且限制并行度，必要时强制 CPU，界面与特效精简/关闭；可用内存极低时仍会尝试本地推理（强制 CPU/收缩上下文），失败或缺模型才转向远程/爬虫补强。

## 测试
运行脚本后生成的 `test_report.json` 会包含硬件等级、模型响应长度、存续阈值等信息；详细运行日志位于 `logs/`。

## 注意事项
- 未安装 Ollama 或模型时会以中文日志提示并尽量保持程序存活。
- 旧版 Ollama CLI 可能不支持 `--num-thread/--num-threads` 参数，代码已改为仅用环境变量（如 `OLLAMA_NUM_CTX`、`OLLAMA_NUM_THREADS`）传递上下文与线程设置，避免“unknown flag”导致推理失败。
- 加密/备份逻辑为占位实现，请按实际需求加固。
- 服务器无界面场景下可在首次运行后将 `config.json` 的 `ui_enabled` 设置为 `false`。
- 若长时间未弹出窗口，请检查控制台是否出现“未检测到 PyQt6”或“创建 Qt 应用失败”提示；若 PyQt6 已安装且日志提示已自动开启窗口模式，仍无界面时请确认在有桌面的环境运行。
- 在 Linux 无图形会话（缺少 `DISPLAY` 环境变量）时，程序会提示窗口可能无法弹出，此时可改为图形桌面环境或将 `ui_enabled` 设为 `false` 进入无界面模式；关闭窗口后的 Qt 退出码会在日志中记录，便于排查异常退出。
- 若本地模型检查失败会进入 30 秒冷却，避免频繁 `ollama list` 卡顿；冷却期间优先远程推理，远程超时会提前切源并继续爬取。
- Windows 控制台默认使用本地编码（如 GBK），代码已在模型调用与远程爬取中加入 UTF-8 宽松解码并忽略异常字符，仍出现乱码可先执行 `chcp 65001`。

## 文件结构
- `ai_humanoid.py` —— 主要代码与入口。
- `config.json` —— 运行时生成的配置文件。
- `logs/` —— 运行日志与复盘记录。
- `test_report.json` —— 每次运行后生成的自检报告。
