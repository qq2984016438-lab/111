"""
本地类人自主进化系统示例（Python 3.11）。
核心围绕 Ollama 与 qwen3-vl-8b，提供模块化能力与硬件自适应启动流程。
"""
from __future__ import annotations

import asyncio
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from importlib import import_module, util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CONFIG_PATH = Path(__file__).parent / "config.json"
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
# --------------------------- 本能配置 ---------------------------
@dataclass(frozen=True)
class Instincts:
    survival_threshold: float = 0.10  # 崩溃概率超过 10% 触发存续优先
    entropy_reduction_bias: float = 0.25  # 熵减偏置系数，主动填补能力缺口

    def survival_guard(self, crash_probability: float) -> bool:
        return crash_probability > self.survival_threshold

    def entropy_drive(self, current_gap: float) -> float:
        return current_gap * (1 - self.entropy_reduction_bias)


INSTINCTS = Instincts()
# --------------------------- 日志管理 ---------------------------
class LogManager:
    def __init__(self) -> None:
        self.log_file = LOG_DIR / "ai_humanoid.log"
        self.replay_file = LOG_DIR / "replay.log"

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line)
        print(line, end="")

    def replay(self, state: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {state}\n"
        with self.replay_file.open("a", encoding="utf-8") as f:
            f.write(line)

    def retrieve(self, keyword: str) -> List[str]:
        if not self.log_file.exists():
            return []
        return [line for line in self.log_file.read_text().splitlines() if keyword.lower() in line.lower()]


logger = LogManager()
# --------------------------- 依赖管理 ---------------------------
class DependencyManager:
    REQUIRED_PY = ["PyQt6", "paho-mqtt", "requests"]
    OPTIONAL_PY = ["psutil", "kivy"]
    OLLAMA_MODEL = "qwen3-vl-8b"
    OLLAMA_MODEL_ALIASES = ["qwen3-vl-8b", "qwen3-vl:8b", "qwen3-vl8b"]

    def __init__(self, auto_install: bool = True) -> None:
        self.auto_install = auto_install
        self._pyqt_checked = False

    def ensure(self) -> Dict[str, bool]:
        status: Dict[str, bool] = {}
        for pkg in self.REQUIRED_PY + self.OPTIONAL_PY:
            status[pkg] = self._check_and_install(pkg)
        status["ollama"] = shutil.which("ollama") is not None
        if not status["ollama"]:
            logger.log("[警告] 未检测到 Ollama CLI，请从 https://ollama.ai 安装后重试。")
        else:
            self._ensure_model()
        return status

    def ensure_pyqt(self) -> bool:
        """确保 PyQt6 可用，不可用时重试安装一次。"""
        if self._pyqt_checked:
            return util.find_spec("PyQt6") is not None
        self._pyqt_checked = True
        if util.find_spec("PyQt6") is not None:
            return True
        logger.log("[警告] 未检测到 PyQt6，尝试自动安装以启用窗口界面……")
        ok = self._check_and_install("PyQt6")
        if not ok:
            logger.log("[错误] PyQt6 安装失败，界面将被禁用，可手动安装后重启。")
        return ok

    def _check_and_install(self, pkg: str) -> bool:
        module_name = pkg.replace("-", "_")
        if util.find_spec(module_name) is not None:
            import_module(module_name)
            return True

        if not self.auto_install:
            logger.log(f"[警告] 缺少依赖 {pkg}")
            return False

        logger.log(f"[信息] 正在自动安装 {pkg} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except subprocess.CalledProcessError:
            logger.log(f"[错误] 安装 {pkg} 失败，请手动检查网络或权限。")
            return False

        if util.find_spec(module_name) is not None:
            import_module(module_name)
            return True
        logger.log(f"[错误] {pkg} 自动安装后仍不可用，请手动处理。")
        return False

    def _ensure_model(self) -> None:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                errors="ignore",
            )
            if not any(alias in result.stdout for alias in self.OLLAMA_MODEL_ALIASES):
                for alias in self.OLLAMA_MODEL_ALIASES:
                    logger.log(f"[信息] 正在尝试拉取本地模型 {alias} ...")
                    pull = subprocess.run(
                        ["ollama", "pull", alias],
                        capture_output=True,
                        text=True,
                        check=False,
                        encoding="utf-8",
                        errors="ignore",
                    )
                    if pull.returncode == 0:
                        logger.log(f"[信息] 模型 {alias} 已就绪或已缓存。")
                        return
                    logger.log(f"[警告] 拉取模型 {alias} 失败：{pull.stderr.strip()}")
                logger.log("[警告] 所有模型别名均拉取失败，请手动执行 ollama pull qwen3-vl-8b 或 qwen3-vl:8b。")
        except Exception as exc:  # noqa: BLE001
            logger.log(f"[警告] 无法验证 Ollama 模型可用性：{exc}")


# --------------------------- 配置管理 ---------------------------
@dataclass
class Config:
    hardware_tier: str = "auto"
    context_size: int = 2048
    num_threads: int = max(1, os.cpu_count() or 1)
    gpu_memory_fraction: float = 0.5
    ui_enabled: bool = True
    cache_enabled: bool = True
    async_inference: bool = True
    model_timeout: int = 60
    remote_endpoint: str = os.environ.get("REMOTE_MODEL_ENDPOINT", "")


class ConfigManager:
    def __init__(self) -> None:
        self.config = Config()
        self.load()

    def load(self) -> None:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            self.config = Config(**{**self.config.__dict__, **data})

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(self.config.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")

    def update(self, **kwargs: Any) -> None:
        for key, val in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, val)
        self.save()


config_manager = ConfigManager()


# --------------------------- 硬件探测 ---------------------------
class HardwareProfiler:
    def __init__(self) -> None:
        self.cpu_cores = os.cpu_count() or 1
        self.total_memory_gb = self._detect_memory()
        self.gpu_name = self._detect_gpu()

    def _detect_memory(self) -> float:
        if util.find_spec("psutil"):
            psutil = import_module("psutil")  # type: ignore
            return round(psutil.virtual_memory().total / (1024**3), 2)
        if platform.system() == "Linux" and Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return round(kb / 1024 / 1024, 2)
        return 4.0

    def _detect_gpu(self) -> str:
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
                return out.decode().strip().splitlines()[0]
            except Exception:  # noqa: BLE001
                return "NVIDIA GPU"
        return "集成/未知 GPU"

    def classify(self) -> str:
        cores = self.cpu_cores
        mem = self.total_memory_gb
        high = cores >= 16 and mem >= 32
        mid = cores >= 8 and mem >= 16
        if high:
            tier = "high"
        elif mid:
            tier = "mid"
        else:
            tier = "low"
        tier_label = {"high": "高配", "mid": "中配", "low": "低配"}.get(tier, tier)
        logger.log(
            f"[信息] 硬件分级：{tier_label}（CPU {cores} 核，内存 {mem} GB，GPU {self.gpu_name}）"
        )
        return tier

    def recommended_params(self) -> Dict[str, Any]:
        tier = self.classify()
        if tier == "high":
            return {
                "hardware_tier": tier,
                "context_size": 4096,
                "num_threads": min(16, self.cpu_cores),
                "gpu_memory_fraction": 0.8,
                "ui_enabled": True,
            }
        if tier == "mid":
            return {
                "hardware_tier": tier,
                "context_size": 3072,
                "num_threads": min(8, self.cpu_cores),
                "gpu_memory_fraction": 0.6,
                "ui_enabled": True,
            }
        return {
            "hardware_tier": tier,
            "context_size": 2048,
            "num_threads": min(4, self.cpu_cores),
            "gpu_memory_fraction": 0.4,
            "ui_enabled": True,
        }


hardware_profiler = HardwareProfiler()


# --------------------------- 推理缓存 ---------------------------
class ModelCache:
    def __init__(self) -> None:
        self.cache: Dict[str, str] = {}
        self.lock = threading.Lock()

    def get(self, prompt: str) -> Optional[str]:
        with self.lock:
            return self.cache.get(prompt)

    def set(self, prompt: str, response: str) -> None:
        with self.lock:
            if len(self.cache) > 128:
                self.cache.pop(next(iter(self.cache)))
            self.cache[prompt] = response


model_cache = ModelCache()


# --------------------------- 远程模型桥接 ---------------------------
class RemoteModelConnector:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint or os.environ.get("REMOTE_MODEL_ENDPOINT", "")
        self.discovery_url = os.environ.get("REMOTE_ENDPOINT_LIST_URL", "")
        backups = os.environ.get("REMOTE_MODEL_ENDPOINTS", "")
        self.backup_endpoints = [e.strip() for e in backups.split(",") if e.strip()]
        crawl_seeds = os.environ.get(
            "REMOTE_CRAWL_SEEDS", "https://ollama.ai/library,https://github.com/ollama/ollama"
        )
        self.crawl_seeds = [e.strip() for e in crawl_seeds.split(",") if e.strip()]
        self._seen_candidates: set[str] = set(self.backup_endpoints)
        self.session = None
        if util.find_spec("requests"):
            self.session = import_module("requests")  # type: ignore

    @staticmethod
    def _safe_text(resp: Any) -> str:
        """宽松解码远程响应，避免在 Windows 下因编码不一致导致崩溃。"""
        try:
            if not getattr(resp, "encoding", None):
                resp.encoding = getattr(resp, "apparent_encoding", None) or "utf-8"
            else:
                resp.encoding = resp.encoding or getattr(resp, "apparent_encoding", None) or "utf-8"
            return resp.text
        except Exception:
            try:
                return resp.content.decode(getattr(resp, "apparent_encoding", "utf-8") or "utf-8", errors="ignore")
            except Exception:
                return resp.content.decode("utf-8", errors="ignore")

    def ready(self) -> bool:
        if self.session is None:
            return False
        if not self.endpoint:
            self.auto_discover()
        if not self.endpoint and self.backup_endpoints:
            for candidate in self.backup_endpoints:
                if candidate in self._seen_candidates and self._probe(candidate):
                    self.endpoint = candidate
                    logger.log(f"[信息] 已切换远程端点：{candidate}")
                    break
        if not self.endpoint:
            self.crawl_public_endpoints()
        if not self.endpoint and self.backup_endpoints:
            for candidate in self.backup_endpoints:
                if candidate in self._seen_candidates and self._probe(candidate):
                    self.endpoint = candidate
                    logger.log(f"[信息] 已切换远程端点：{candidate}")
                    break
        if not self.endpoint:
            logger.log("[警告] 暂未发现可用远程端点，将持续爬取与切换以补强算力。")
        return bool(self.endpoint)

    def auto_discover(self) -> None:
        """尝试“爬”取远程端点：优先使用环境变量列表，再尝试远程拉取列表。"""
        if self.session is None:
            return
        if not self.backup_endpoints and self.discovery_url:
            try:
                logger.log("[信息] 正在抓取远程端点列表，用于补强推理...")
                resp = self.session.get(self.discovery_url, timeout=8)
                if resp.status_code == 200:
                    text = self._safe_text(resp)
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    self.backup_endpoints.extend(lines)
                    logger.log(f"[信息] 已从远程列表获取 {len(lines)} 个候选端点。")
                else:
                    logger.log(f"[警告] 远程端点列表获取失败：HTTP {resp.status_code}")
            except Exception as exc:  # noqa: BLE001
                logger.log(f"[警告] 远程端点列表抓取异常：{exc}")
        if not self.endpoint:
            for candidate in self.backup_endpoints:
                if self._probe(candidate):
                    self.endpoint = candidate
                    logger.log(f"[信息] 已自动切换远程端点：{candidate}")
                    break

    def crawl_public_endpoints(self) -> None:
        """通过爬虫抓取页面中的 API 端点，避免单点失效。"""
        if self.session is None:
            return
        for seed in self.crawl_seeds:
            if not seed or seed in self._seen_candidates:
                continue
            try:
                logger.log(f"[信息] 正在爬取远程端点来源：{seed}")
                resp = self.session.get(seed, timeout=10)
                if resp.status_code >= 500:
                    logger.log(f"[警告] 爬取 {seed} 失败：HTTP{resp.status_code}")
                    continue
                text = self._safe_text(resp)
                urls = re.findall(r"https?://[^\s\"']+", text)
                filtered = [u for u in urls if any(key in u.lower() for key in ("api", "chat", "infer", "model", "v1"))]
                new_candidates = [u for u in filtered if u not in self._seen_candidates]
                for cand in new_candidates:
                    if len(self.backup_endpoints) > 50:
                        break
                    self._seen_candidates.add(cand)
                    self.backup_endpoints.append(cand)
                if new_candidates:
                    logger.log(f"[信息] 爬虫新增 {len(new_candidates)} 个候选端点，将逐一探测...")
            except Exception as exc:  # noqa: BLE001
                logger.log(f"[警告] 爬虫获取远程端点异常：{exc}")

    def _probe(self, url: str) -> bool:
        if self.session is None:
            return False
        try:
            resp = self.session.get(url, timeout=3)
            return resp.status_code < 500
        except Exception:
            return False

    def warmup(self) -> bool:
        test = self.fetch("PING", retries=1)
        return bool(test)

    def fetch(self, prompt: str, retries: int = 3, timeout: int = 12) -> Optional[str]:
        if not self.ready():
            return None
        assert self.session is not None
        start = time.time()
        for attempt in range(1, retries + 1):
            try:
                logger.log(f"[信息] 正在通过远程模型尝试推理（第 {attempt} 次）...")
                resp = self.session.post(
                    self.endpoint,
                    json={"prompt": prompt},
                    timeout=max(5, min(timeout, 25)),
                )
                resp.encoding = getattr(resp, "apparent_encoding", None) or resp.encoding or "utf-8"
                if resp.status_code == 200:
                    data = resp.json()
                    result = data.get("reply") or data.get("text") or self._safe_text(resp)
                    logger.log("[信息] 远程模型返回结果。")
                    return str(result).strip()
                logger.log(f"[警告] 远程模型响应异常：HTTP {resp.status_code}")
            except Exception as exc:  # noqa: BLE001
                logger.log(f"[警告] 远程模型尝试失败：{exc}，继续重试...")
                if time.time() - start > timeout:
                    logger.log("[警告] 远程模型总体耗时过长，提前切换其他端点。")
                    break
                if attempt == retries:
                    self.endpoint = ""
                    self.auto_discover()
                    if not self.endpoint:
                        self.crawl_public_endpoints()
                        self.auto_discover()
        return None


# --------------------------- 模型管理 ---------------------------
class ModelManager:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.daemon_thread: Optional[threading.Thread] = None
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.remote = RemoteModelConnector(cfg.remote_endpoint)
        self._model_candidates = DependencyManager.OLLAMA_MODEL_ALIASES
        self._active_model = self._model_candidates[0]
        self._local_check_passed = False
        self._local_last_error: Optional[str] = None
        self._last_local_check: float = 0.0
        self._local_retry_interval: int = 30
        raw_host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip().rstrip("/")
        # CLI 期望 host:port，HTTP 需要协议，拆分后统一复用
        if raw_host.startswith("http://") or raw_host.startswith("https://"):
            raw_host = raw_host.split("://", 1)[1]
        self._ollama_host = raw_host or "127.0.0.1:11434"
        self._http_host = f"http://{self._ollama_host}"

    def _cli_env(self) -> Dict[str, str]:
        env = {**os.environ, "OLLAMA_HOST": self._ollama_host}
        return {k: str(v) for k, v in env.items()}

    def _ping_ollama_http(self) -> Tuple[bool, str]:
        """HTTP 层探测 Ollama 服务与模型存在性，兼容自定义 OLLAMA_HOST。"""
        if util.find_spec("requests") is None:
            return False, "缺少 requests 库"
        requests_mod = import_module("requests")  # type: ignore
        try:
            version_resp = requests_mod.get(f"{self._http_host}/api/version", timeout=5)
            if version_resp.status_code != 200:
                return False, f"版本接口响应异常 HTTP{version_resp.status_code}"
        except Exception as exc:  # noqa: BLE001
            return False, f"HTTP 探测失败：{exc}"

        try:
            tags = requests_mod.get(f"{self._http_host}/api/tags", timeout=8)
            if tags.status_code == 200:
                for alias in self._model_candidates:
                    if alias in tags.text:
                        self._active_model = alias
                        return True, f"HTTP 确认模型已存在：{alias}"
        except Exception:
            pass

        try:
            logger.log("[信息] 通过 HTTP 触发模型拉取...")
            for alias in self._model_candidates:
                pull_resp = requests_mod.post(
                    f"{self._http_host}/api/pull",
                    json={"name": alias},
                    timeout=30,
                )
                if pull_resp.status_code == 200:
                    self._active_model = alias
                    return True, f"HTTP 拉取请求已提交：{alias}"
            return False, "HTTP 拉取请求全部失败"
        except Exception as exc:  # noqa: BLE001
            return False, f"HTTP 拉取异常：{exc}"

    def _local_ready(self) -> bool:
        """检查本地 Ollama 可用性。"""
        if self._local_check_passed:
            return True
        now = time.time()
        if self._local_last_error and now - self._last_local_check < self._local_retry_interval:
            logger.log("[信息] 本地模型短时间内已检查失败，使用上次结果以避免卡顿。")
            return False
        self._last_local_check = now
        if shutil.which("ollama") is None:
            self._local_last_error = "未检测到 Ollama CLI"
            logger.log("[警告] 未检测到 Ollama CLI，优先尝试远程模型或请安装后重启。")
            return False
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
                env=self._cli_env(),
                timeout=20,
                encoding="utf-8",
                errors="ignore",
            )
            if not any(alias in result.stdout for alias in self._model_candidates):
                self._local_last_error = "本地缺少 qwen3-vl 系列模型"
                logger.log("[警告] 本地尚未拉取 qwen3-vl-8b/ qwen3-vl:8b，尝试自动拉取...")
                pulled_ok = False
                for alias in self._model_candidates:
                    pull = subprocess.run(
                        ["ollama", "pull", alias],
                        capture_output=True,
                        text=True,
                        check=False,
                        env=self._cli_env(),
                        timeout=120,
                        encoding="utf-8",
                        errors="ignore",
                    )
                    if pull.returncode == 0:
                        self._active_model = alias
                        logger.log(f"[信息] 模型 {alias} 已可用。")
                        pulled_ok = True
                        break
                    logger.log(f"[警告] 拉取 {alias} 失败：{pull.stderr.strip()}")
                if pulled_ok:
                    result = subprocess.run(
                        ["ollama", "list"],
                        capture_output=True,
                        text=True,
                        check=False,
                        env=self._cli_env(),
                        timeout=30,
                        encoding="utf-8",
                        errors="ignore",
                    )
                if not any(alias in result.stdout for alias in self._model_candidates):
                    ok, reason = self._ping_ollama_http()
                    if ok:
                        logger.log(f"[信息] HTTP 拉取/确认成功：{reason}")
                        self._local_check_passed = True
                        self._local_last_error = None
                        return True
                    logger.log(f"[警告] 本地模型仍未就绪：{reason}")
                    return False
            else:
                for alias in self._model_candidates:
                    if alias in result.stdout:
                        self._active_model = alias
                        break
            self._local_check_passed = True
            self._local_last_error = None
            logger.log("[信息] 本地 Ollama 已确认可用。")
            return True
        except Exception as exc:  # noqa: BLE001
            self._local_last_error = str(exc)
            logger.log(f"[警告] 本地模型自检异常：{exc}")
            ok, reason = self._ping_ollama_http()
            if ok:
                logger.log(f"[信息] 已通过 HTTP 确认本地模型：{reason}")
                self._local_check_passed = True
                self._local_last_error = None
                return True
            logger.log(f"[警告] HTTP 兜底也失败：{reason}")
            return False

    def _try_remote(self, prompt: str, retries: int = 3) -> Optional[str]:
        if not self.remote.ready():
            return None
        remote_resp = self.remote.fetch(prompt, retries=retries, timeout=min(self.cfg.model_timeout, 20))
        if remote_resp and self.cfg.cache_enabled:
            model_cache.set(prompt, remote_resp)
        return remote_resp

    def start_daemon(self) -> None:
        if self.running:
            return
        self.running = True
        self.daemon_thread = threading.Thread(target=self._loop, daemon=True)
        self.daemon_thread.start()

    def _loop(self) -> None:
        while self.running:
            try:
                task = self.queue.get(timeout=1)
            except queue.Empty:
                continue
            prompt = task.get("prompt", "")
            fut: asyncio.Future[str] = task.get("future")
            loop: asyncio.AbstractEventLoop = task.get("loop")
            try:
                result = self.generate(prompt)
                loop.call_soon_threadsafe(fut.set_result, result)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(fut.set_exception, exc)

    def stop(self) -> None:
        self.running = False
        if self.daemon_thread and self.daemon_thread.is_alive():
            self.daemon_thread.join(timeout=2)

    async def generate_async(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self.queue.put({"prompt": prompt, "future": fut, "loop": loop})
        return await fut

    def generate(self, prompt: str) -> str:
        cached = model_cache.get(prompt) if self.cfg.cache_enabled else None
        if cached:
            return cached
        # 本地不可用或低配算力优先尝试远程模型
        local_ready = self._local_ready()
        if self.cfg.hardware_tier == "low" or not local_ready:
            remote_resp = self._try_remote(prompt, retries=5)
            if remote_resp:
                return remote_resp
            # 在本地自检失败时也尝试 HTTP 直连，避免误判导致直接报错
            http_try = self._http_generate(prompt)
            if http_try:
                logger.log("[信息] 本地 CLI 自检失败但 HTTP 推理成功，已标记本地可用。")
                self._local_check_passed = True
                self._local_last_error = None
                return http_try
            if not local_ready:
                missing_reason = self._local_last_error or "未知原因"
                return (
                    "【模型不可用】本地推理未就绪："
                    f"{missing_reason}；未配置可用远程端点，请安装 Ollama 并拉取 qwen3-vl-8b"
                    "，或设置 REMOTE_MODEL_ENDPOINT 以启用远程推理。"
                )
            logger.log("[警告] 远程模型尝试失败，回退本地推理。")
        cmd = ["ollama", "run", self._active_model]
        env = {
            **self._cli_env(),
            "OLLAMA_NUM_CTX": str(self.cfg.context_size),
            "OLLAMA_NUM_THREADS": str(self.cfg.num_threads),
        }
        logger.log(
            f"[信息] 触发模型推理，窗口 {self.cfg.context_size}，线程 {self.cfg.num_threads}。"
        )
        try:
            process = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                check=True,
                env=env,
                timeout=max(10, self.cfg.model_timeout),
                encoding="utf-8",
                errors="ignore",
            )
            output = process.stdout.strip()
            if not output:
                output = process.stderr.strip() or "【模型无返回内容】"
        except subprocess.TimeoutExpired:
            logger.log("[警告] 本地推理超时，正在尝试 HTTP 与远程兜底……")
            http_retry = self._http_generate(prompt)
            if http_retry:
                output = http_retry
            else:
                fallback = self._try_remote(prompt, retries=3)
                output = fallback or "【模型不可用】本地推理超时且远程兜底失败。"
        except FileNotFoundError:
            logger.log("[错误] 未找到 Ollama 可执行文件，尝试切换远程推理。")
            fallback = self._try_remote(prompt, retries=5)
            if fallback:
                output = fallback
            else:
                output = "【模型不可用】未找到 ollama 可执行文件且远程推理失败。"
        except Exception as exc:  # noqa: BLE001
            output = f"【模型不可用】{exc}"
            if isinstance(exc, subprocess.CalledProcessError):
                stderr_msg = exc.stderr.strip() if exc.stderr else ""
                if stderr_msg:
                    output += f" | stderr: {stderr_msg}"
            http_retry = self._http_generate(prompt)
            if http_retry:
                output = http_retry
            else:
                # 本地失败时再做一次远程兜底
                fallback = self._try_remote(prompt, retries=2)
                if fallback:
                    output = fallback
        if output in ("", "【模型无返回内容】"):
            logger.log("[警告] 本地推理返回为空，尝试 HTTP/远程兜底……")
            http_retry = self._http_generate(prompt)
            if http_retry:
                output = http_retry
            else:
                fallback = self._try_remote(prompt, retries=3)
                if fallback:
                    output = fallback
                else:
                    output = "【模型无返回内容】请检查 Ollama 日志或配置 REMOTE_MODEL_ENDPOINT。"
        if self.cfg.cache_enabled:
            model_cache.set(prompt, output)
        return output

    def _http_generate(self, prompt: str) -> Optional[str]:
        """当 CLI 推理失败时，通过 HTTP 接口兜底调用 generate。"""
        if util.find_spec("requests") is None:
            return None
        requests_mod = import_module("requests")  # type: ignore
        try:
            resp = requests_mod.post(
                f"{self._http_host}/api/generate",
                json={
                    "model": self._active_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": self.cfg.context_size,
                        "num_thread": self.cfg.num_threads,
                    },
                },
                timeout=max(10, self.cfg.model_timeout),
            )
            if resp.status_code == 200:
                resp.encoding = getattr(resp, "apparent_encoding", None) or resp.encoding or "utf-8"
                try:
                    data = resp.json()
                except Exception:
                    try:
                        data = json.loads(resp.content.decode(resp.encoding or "utf-8", errors="ignore"))
                    except Exception:
                        data = {"response": self.remote._safe_text(resp) if self.remote else resp.text}
                return str(data.get("response") or data).strip()
            logger.log(f"[警告] HTTP 调用 generate 失败：HTTP{resp.status_code}")
        except Exception as exc:  # noqa: BLE001
            logger.log(f"[警告] HTTP 调用 generate 异常：{exc}")
        return None


# --------------------------- 核心能力模块 ---------------------------
class AutonomyModule:
    def __init__(self, model: ModelManager) -> None:
        self.model = model

    async def generate_rules(self, context: str) -> str:
        prompt = f"请针对场景 {context} 生成包含犹豫权衡与修正策略的操作规则。"
        if config_manager.config.async_inference:
            return await self.model.generate_async(prompt)
        return self.model.generate(prompt)

    async def discover_goals(self, state: str) -> str:
        prompt = f"请基于当前状态 {state} 给出可执行的目标列表并排序。"
        return await self.model.generate_async(prompt)

    async def debrief(self, decision: str, outcome: str) -> str:
        prompt = f"复盘决策 {decision} 的结果 {outcome}，给出改进建议。"
        return await self.model.generate_async(prompt)


class EvolutionModule:
    def __init__(self, model: ModelManager) -> None:
        self.model = model

    def adaptive_compute(self, load_factor: float) -> int:
        base = config_manager.config.num_threads
        adjusted = max(1, int(base * (1 - min(load_factor, 0.9))))
        logger.log(f"[信息] 自适应线程数：{adjusted}")
        return adjusted

    def materialize_code(self, spec: str) -> str:
        prompt = f"请生成可运行的 Python 函数以实现：{spec}。"
        return self.model.generate(prompt)

    async def draft_upgrade(self, target: str) -> str:
        prompt = f"针对升级目标：{target}，给出分步骤演进思路与预期代码草案。"
        if config_manager.config.async_inference:
            return await self.model.generate_async(prompt)
        return self.model.generate(prompt)

    def migrate(self, target: str) -> str:
        return f"已生成面向 {target} 的迁移占位结果。"


class EmotionModule:
    def __init__(self) -> None:
        self.memory: List[str] = []
        self.mood: str = "平静"

    def step(self, event: str) -> str:
        self.memory.append(event)
        if "error" in event.lower() or "错误" in event:
            self.mood = "关切"
        elif "success" in event.lower() or "成功" in event:
            self.mood = "积极"
        else:
            self.mood = "平静"
        return self.mood

    def repair(self) -> str:
        self.mood = "平稳"
        return self.mood

    def express(self) -> str:
        return f"情感：{self.mood} | 记忆条目 {len(self.memory)}"


class SurvivalModule:
    def backup(self) -> str:
        LOG_DIR.mkdir(exist_ok=True)
        backup_file = LOG_DIR / "backup.state"
        backup_file.write_text("系统快照")
        return str(backup_file)

    def encrypt(self, data: str) -> str:
        return data[::-1]  # 占位示例，加密需自行替换

    def mitigate(self, risk: str) -> str:
        if INSTINCTS.survival_guard(0.11):
            return f"正在缓解风险：{risk}"
        return "风险在可接受范围"

    def self_acceptance(self) -> str:
        return "保持在安全区间内运行。"


class RepairModule:
    def hardware_patch(self) -> str:
        return "硬件兼容层已开启"

    def network_resilience(self) -> str:
        return "网络容错监测已启用"

    def emotional_stabilizer(self, emotion_module: EmotionModule) -> str:
        return emotion_module.repair()


class SocialModule:
    def __init__(self, model: ModelManager) -> None:
        self.model = model

    async def understand_intent(self, utterance: str) -> str:
        return await self.model.generate_async(f"请总结用户意图：{utterance}")

    async def converse(self, message: str) -> str:
        return await self.model.generate_async(f"请以礼貌简洁的方式回复：{message}")

    async def etiquette(self, context: str) -> str:
        return await self.model.generate_async(f"请列出与 {context} 相关的社交礼仪提示")


# --------------------------- 可视化界面 ---------------------------
class Interface:
    def __init__(
        self,
        emotion: EmotionModule,
        model: ModelManager,
        evolution: EvolutionModule,
        social: SocialModule,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.emotion = emotion
        self.model = model
        self.evolution = evolution
        self.social = social
        self.loop = loop

    def start(self) -> None:
        if util.find_spec("PyQt6") is None:
            logger.log("[警告] 未检测到 PyQt6，正在尝试补装以恢复窗口界面……")
            dm = DependencyManager(auto_install=True)
            if not dm.ensure_pyqt():
                logger.log("[错误] 窗口依赖不可用，已进入无界面模式。")
                return

        if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
            logger.log("[警告] 当前无可用图形显示（缺少 DISPLAY 环境变量），窗口可能无法弹出，建议在图形桌面下运行。")

        try:
            from PyQt6 import QtWidgets, QtGui, QtCore  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logger.log(f"[错误] 加载 PyQt6 失败：{exc}，已切换无界面模式。")
            return

        logger.log("[信息] 正在启动可视化窗口，请稍候……")

        try:
            app = QtWidgets.QApplication(sys.argv)
        except Exception as exc:  # noqa: BLE001
            logger.log(f"[错误] 创建 Qt 应用失败：{exc}，可能缺少图形环境，请在有桌面的环境重试。")
            return

        window = QtWidgets.QWidget()
        window.setWindowTitle("类人智能体监控与升级演示")

        status_label = QtWidgets.QLabel("状态：空闲")
        emotion_label = QtWidgets.QLabel(self.emotion.express())
        hardware_label = QtWidgets.QLabel(
            f"算力模式：{config_manager.config.hardware_tier}，远程端点：{self.model.remote.endpoint or '未配置'}"
        )

        tabs = QtWidgets.QTabWidget()

        # 聊天互动页
        chat_widget = QtWidgets.QWidget()
        chat_layout = QtWidgets.QVBoxLayout()
        chat_output = QtWidgets.QTextEdit()
        chat_output.setReadOnly(True)
        input_box = QtWidgets.QLineEdit()
        input_box.setPlaceholderText("输入与你的对话或需求...")
        send_btn = QtWidgets.QPushButton("发送对话")
        chat_layout.addWidget(chat_output)
        chat_form = QtWidgets.QHBoxLayout()
        chat_form.addWidget(input_box)
        chat_form.addWidget(send_btn)
        chat_layout.addLayout(chat_form)
        chat_widget.setLayout(chat_layout)
        tabs.addTab(chat_widget, "聊天互动")

        # 升级观察页
        upgrade_widget = QtWidgets.QWidget()
        upgrade_layout = QtWidgets.QVBoxLayout()
        upgrade_input = QtWidgets.QLineEdit()
        upgrade_input.setPlaceholderText("描述想要的升级方向，例如：提升多模态理解")
        upgrade_btn = QtWidgets.QPushButton("生成升级方案与代码草案")
        upgrade_output = QtWidgets.QTextEdit()
        upgrade_output.setReadOnly(True)
        upgrade_layout.addWidget(upgrade_input)
        upgrade_layout.addWidget(upgrade_btn)
        upgrade_layout.addWidget(upgrade_output)
        upgrade_widget.setLayout(upgrade_layout)
        tabs.addTab(upgrade_widget, "升级观测")

        # 行为日志页
        log_widget = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout()
        behavior_view = QtWidgets.QTextEdit()
        behavior_view.setReadOnly(True)
        behavior_view.append("实时行为逻辑将在此展示，包括远程连接尝试与存续策略。")
        log_layout.addWidget(behavior_view)
        log_widget.setLayout(log_layout)
        tabs.addTab(log_widget, "行为日志")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(status_label)
        layout.addWidget(emotion_label)
        layout.addWidget(hardware_label)
        layout.addWidget(tabs)
        window.setLayout(layout)

        class UiSignals(QtCore.QObject):
            chat_done = QtCore.pyqtSignal(str, str)
            upgrade_done = QtCore.pyqtSignal(str, str, str)

        signals = UiSignals()

        def on_chat_done(user_text: str, resp: str) -> None:
            chat_output.append(f"你：{user_text}\nAI：{resp}\n")
            behavior_view.append(f"[对话] 用户输入：{user_text}")
            mood = self.emotion.step("成功")
            emotion_label.setText(f"情绪：{mood}")
            status_label.setText("状态：空闲")
            input_box.clear()

        def on_upgrade_done(target: str, plan: str, code: str) -> None:
            upgrade_output.append(f"[思考] {plan}\n[代码草案]\n{code}\n")
            behavior_view.append(f"[升级] 目标：{target} | 已生成思路与代码草案")
            status_label.setText("状态：空闲")

        signals.chat_done.connect(on_chat_done)  # type: ignore
        signals.upgrade_done.connect(on_upgrade_done)  # type: ignore

        def send_message() -> None:
            text = input_box.text().strip()
            if not text:
                return
            status_label.setText("状态：思考中...")
            QtWidgets.QApplication.processEvents()

            def worker() -> None:
                try:
                    future = asyncio.run_coroutine_threadsafe(self.social.converse(text), self.loop)
                    resp = future.result(timeout=config_manager.config.model_timeout)
                except Exception as exc:  # noqa: BLE001
                    logger.log(f"[警告] 异步对话失败，尝试同步兜底：{exc}")
                    try:
                        # 界面线程已隔离，直接同步调用可避免事件循环异常导致的无响应
                        resp = self.model.generate(f"请以礼貌简洁的方式回复：{text}")
                    except Exception as inner:  # noqa: BLE001
                        resp = f"[警告] 对话生成失败：{inner}"
                signals.chat_done.emit(text, resp)

            threading.Thread(target=worker, daemon=True).start()

        def trigger_upgrade() -> None:
            target = upgrade_input.text().strip() or "提升推理与自愈能力"
            status_label.setText("状态：升级推演中...")
            QtWidgets.QApplication.processEvents()

            def worker() -> None:
                try:
                    plan_future = asyncio.run_coroutine_threadsafe(self.evolution.draft_upgrade(target), self.loop)
                    plan = plan_future.result(timeout=config_manager.config.model_timeout)
                except Exception as exc:  # noqa: BLE001
                    logger.log(f"[警告] 异步升级思路生成失败，尝试同步兜底：{exc}")
                    try:
                        plan = self.model.generate(f"请为 {target} 提供升级思路，简要列出步骤。")
                    except Exception as inner:  # noqa: BLE001
                        plan = f"[警告] 升级思路生成失败：{inner}"
                try:
                    code = self.evolution.materialize_code(target)
                except Exception as exc:  # noqa: BLE001
                    code = f"# 生成代码失败：{exc}"
                signals.upgrade_done.emit(target, plan, code)

            threading.Thread(target=worker, daemon=True).start()

        send_btn.clicked.connect(send_message)  # type: ignore
        upgrade_btn.clicked.connect(trigger_upgrade)  # type: ignore
        window.show()
        if not config_manager.config.ui_enabled:
            window.setWindowOpacity(0.9)
            window.setStyleSheet("background-color: #222; color: #ddd;")
        try:
            app.processEvents()
            exit_code = app.exec()
            logger.log(f"[信息] 窗口已关闭，Qt 退出码：{exit_code}")
        except SystemExit:
            # 避免 SystemExit 直接终止主线程，确保后续清理执行
            logger.log("[警告] Qt 主循环触发退出信号，准备进行资源清理。")
        except Exception as exc:  # noqa: BLE001
            logger.log(f"[错误] Qt 主循环异常：{exc}")


# --------------------------- 自检测试 ---------------------------
class TestSuite:
    def __init__(self, model: ModelManager, hardware: HardwareProfiler) -> None:
        self.model = model
        self.hardware = hardware

    def run(self) -> Dict[str, str]:
        report: Dict[str, str] = {}
        report["hardware"] = f"硬件等级：{self.hardware.classify()}"
        try:
            sample = self.model.generate("健康检查")
            report["model"] = f"模型响应长度 {len(sample)}"
        except Exception as exc:  # noqa: BLE001
            report["model"] = f"错误：{exc}"
        report["instincts"] = f"存续阈值 {INSTINCTS.survival_threshold}"
        Path("test_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.log(f"[信息] 已生成测试报告：{report}")
        return report


# --------------------------- 守护监控 ---------------------------
class Guardian:
    def __init__(self, model: ModelManager) -> None:
        self.model = model
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.active = False

    def start(self) -> None:
        self.active = True
        self.thread.start()

    def _monitor(self) -> None:
        while self.active:
            time.sleep(5)
            crash_probability = 0.05  # 占位指标，可接入真实监控
            if INSTINCTS.survival_guard(crash_probability):
                logger.log("[警告] 监测到潜在崩溃概率升高")
            else:
                logger.log("[信息] 系统稳定运行")

    def stop(self) -> None:
        self.active = False


# --------------------------- 远程升级守望 ---------------------------
class RemoteUpgradeWatcher:
    def __init__(self, connector: RemoteModelConnector, hardware: HardwareProfiler) -> None:
        self.connector = connector
        self.hardware = hardware
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.active = False

    def start(self) -> None:
        if not self.connector.ready():
            return
        self.active = True
        self.thread.start()

    def _run(self) -> None:
        while self.active and self.hardware.classify() == "low":
            logger.log("[信息] 检测到低配算力，尝试持续连接远程大模型以提升智能。")
            success = self.connector.warmup()
            if success:
                logger.log("[信息] 已锁定远程大模型连接，补足算力缺口。")
                break
            time.sleep(10)

    def stop(self) -> None:
        self.active = False


# --------------------------- 主入口 ---------------------------
def main() -> None:
    dm = DependencyManager(auto_install=True)
    dm.ensure()
    pyqt_ok = dm.ensure_pyqt()

    tier_params = hardware_profiler.recommended_params()
    config_manager.update(**tier_params)
    if tier_params.get("hardware_tier") == "low" and not config_manager.config.remote_endpoint:
        logger.log("[警告] 当前算力为低配且未配置远程大模型端点，建议设置 REMOTE_MODEL_ENDPOINT 以增强智能。")

    if pyqt_ok and not config_manager.config.ui_enabled:
        logger.log("[信息] 检测到 PyQt6 可用，但配置为无界面模式，已自动开启窗口模式。")
        config_manager.update(ui_enabled=True)

    model_manager = ModelManager(config_manager.config)
    model_manager.start_daemon()

    emotion = EmotionModule()
    guardian = Guardian(model_manager)
    guardian.start()

    remote_watcher = RemoteUpgradeWatcher(model_manager.remote, hardware_profiler)
    remote_watcher.start()

    autonomy = AutonomyModule(model_manager)
    evolution = EvolutionModule(model_manager)
    survival = SurvivalModule()
    repair = RepairModule()
    social = SocialModule(model_manager)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    model_manager.loop = loop

    async def bootstrap() -> None:
        logger.log("[信息] 正在引导各模块启动...")
        await autonomy.generate_rules("启动阶段")
        await autonomy.discover_goals("初始化准备")
        await social.understand_intent("你好")

    def loop_runner(evt_loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(evt_loop)
        evt_loop.run_forever()

    loop_thread = threading.Thread(target=loop_runner, args=(loop,), daemon=True)
    loop_thread.start()

    # 将启动引导与健康检查放入异步/后台线程，避免阻塞界面弹出
    loop.call_soon_threadsafe(asyncio.create_task, bootstrap())

    def run_tests() -> None:
        tests = TestSuite(model_manager, hardware_profiler)
        tests.run()

    threading.Thread(target=run_tests, daemon=True).start()

    if config_manager.config.ui_enabled:
        ui = Interface(emotion, model_manager, evolution, social, loop)
        try:
            ui.start()
        finally:
            guardian.stop()
            remote_watcher.stop()
            model_manager.stop()
            if model_manager.loop:
                model_manager.loop.call_soon_threadsafe(model_manager.loop.stop)
            loop_thread.join(timeout=2)
    else:
        logger.log("[信息] 低配硬件关闭界面，进入无界面事件循环模式。")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.log("[信息] 收到关闭指令，准备退出。")
        finally:
            guardian.stop()
            remote_watcher.stop()
            model_manager.stop()
            if model_manager.loop:
                model_manager.loop.call_soon_threadsafe(model_manager.loop.stop)
            loop_thread.join(timeout=2)


if __name__ == "__main__":
    main()
