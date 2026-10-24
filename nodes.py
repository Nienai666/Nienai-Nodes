"""
Nienai-Nodes — GrsAI 图像生成节点
===================================
ImageGen / AssetGen / SceneCompositor / Storyboard / ShotFilter / RemoveBG / ScriptExtractor / AudioConcat
"""

import base64
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import requests
from PIL import Image

import comfy.utils
import folder_paths

# ══════════════════════════════════════════════════════════
# 全局常量
# ══════════════════════════════════════════════════════════

GRSAI_BASE_URL = "https://grsai.dakka.com.cn"


def _load_api_key() -> str:
    """读取 GrsAI 的 API Key（不要把密钥写进源码，否则一旦上传就会泄露）。

    读取顺序：
      1. 环境变量 GRSAI_API_KEY；
      2. 本目录下的 .grsai_key 文件（一行明文，已加入 .gitignore，不会上传）；
      3. 都没有 → 返回空字符串，节点上需要手动填写。
    """
    env_key = os.environ.get("GRSAI_API_KEY", "").strip()
    if env_key:
        return env_key
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".grsai_key")
    try:
        with open(key_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


GRSAI_API_KEY = _load_api_key()
IMAGE_MODES = ["generate", "edit"]

MODEL_OPTIONS = [
    "gpt-image-2", "gpt-image-2-vip",
    "nano-banana-2", "nano-banana-2-cl", "nano-banana-2-2k-cl", "nano-banana-2-4k-cl",
]

_WIN_ILLEGAL = re.compile(r'[\\/:*?"<>|]')


# ══════════════════════════════════════════════════════════
# 通用工具函数
# ══════════════════════════════════════════════════════════

def _log(msg):
    print(f"[GrsAI] {msg}", flush=True)


def tensor2pil(t: torch.Tensor) -> Image.Image:
    if len(t.shape) == 4:
        t = t[0]
    arr = np.clip(t.cpu().numpy(), 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil2tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return torch.from_numpy(arr).unsqueeze(0)


def download_image(url: str, timeout: int = 120) -> torch.Tensor:
    _log(f"下载图片: {url[:80]}...")
    resp = requests.get(url, timeout=timeout,
                        headers={"User-Agent": "ComfyUI-GrsAI/1.0"})
    _log(f"下载状态: {resp.status_code}, 大小: {len(resp.content)} bytes")
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content))
    _log(f"图片尺寸: {img.size}, 模式: {img.mode}")
    if img.mode != "RGB":
        img = img.convert("RGB")
    return pil2tensor(img)


def placeholder(w: int = 512, h: int = 512) -> torch.Tensor:
    return torch.from_numpy(np.ones((h, w, 3), dtype=np.float32) * 0.5).unsqueeze(0)


def _pil_to_base64_data_uri(pil_img: Image.Image, fmt: str = "PNG") -> str:
    """PIL Image → "data:image/png;base64,..." 字符串"""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


# ══════════════════════════════════════════════════════════
# 脚本解析器
# ══════════════════════════════════════════════════════════

def _extract_tag(block, tag):
    m = re.search(rf'\[{tag}\]\s*([\s\S]*?)(?=\r?\n\[|\Z)', block)
    return m.group(1).strip() if m else ''


def _extract_free_field(body, label):
    m = re.search(
        rf'{label}[：:；;]\s*([\s\S]*?)(?=\r?\n(?:镜头文案|图片提示词|视频提示词|TTS提示词|景别|语音分镜|音效)[：:；;]|\Z)',
        body
    )
    return m.group(1).strip() if m else ''


def _unify_tensor_sizes(tensors, tag="AssetGen"):
    """确保 batch 中所有 tensor H/W 一致，不一致则 resize 到最大尺寸"""
    if len(tensors) <= 1:
        return tensors
    shapes = set((t.shape[1], t.shape[2]) for t in tensors)
    if len(shapes) <= 1:
        return tensors
    _log(f"[{tag}] 尺寸不一致 {shapes}, 统一 resize...")
    target_h = max(s[0] for s in shapes)
    target_w = max(s[1] for s in shapes)
    resized = []
    for t in tensors:
        if t.shape[1] != target_h or t.shape[2] != target_w:
            t2 = t.permute(0, 3, 1, 2)
            t2 = torch.nn.functional.interpolate(
                t2, size=(target_h, target_w), mode='bilinear', antialias=True)
            t = t2.permute(0, 2, 3, 1)
        resized.append(t)
    return resized


def parse_script_text(text):
    """解析脚本文本 -> { characters, scene, props, shots }"""
    result = {'characters': [], 'scene': None, 'props': [], 'shots': []}

    char_match = re.search(r'\[?角色信息\]?\s*\r?\n([\s\S]*?)(?=\r?\n场景信息|\r?\n道具信息|\r?\n\[镜头\d|\Z)', text)
    if char_match:
        for b in re.split(r'\r?\n(?=\[名称\])', char_match.group(1)):
            if '[名称]' in b and '[描述]' in b:
                result['characters'].append({
                    'name': _extract_tag(b, '名称'),
                    'alias': _extract_tag(b, '别名'),
                    'description': _extract_tag(b, '描述'),
                    'voice': _extract_tag(b, '角色音色'),
                })

    scene_match = re.search(r'场景信息\s*\r?\n([\s\S]*?)(?=\r?\n道具信息|\r?\n角色信息|\r?\n\[镜头\d|\Z)', text)
    if scene_match:
        sb = scene_match.group(1)
        result['scene'] = {
            'name': _extract_tag(sb, '名称'),
            'alias': _extract_tag(sb, '别名'),
            'description': _extract_tag(sb, '描述'),
        }

    prop_match = re.search(r'道具信息\s*\r?\n([\s\S]*?)(?=\r?\n角色信息|\r?\n场景信息|\r?\n\[镜头\d|\Z)', text)
    if prop_match:
        for b in re.split(r'\r?\n(?=\[名称\])', prop_match.group(1)):
            if '[名称]' in b and '[描述]' in b:
                result['props'].append({
                    'name': _extract_tag(b, '名称'),
                    'alias': _extract_tag(b, '别名'),
                    'description': _extract_tag(b, '描述'),
                })

    for m in re.finditer(r'\[?镜头(\d+)\]?\s*\r?\n([\s\S]*?)(?=\r?\n\[?镜头\d|\Z)', text):
        body = m.group(2)
        result['shots'].append({
            'id': int(m.group(1)),
            '镜头文案': _extract_free_field(body, '镜头文案'),
            '图片提示词': _extract_free_field(body, '图片提示词'),
            '视频提示词': _extract_free_field(body, '视频提示词'),
            'TTS提示词': _extract_free_field(body, 'TTS提示词'),
            '景别': _extract_free_field(body, '景别'),
            '语音分镜': _extract_free_field(body, '语音分镜'),
            '音效': _extract_free_field(body, '音效'),
        })

    return result


# ══════════════════════════════════════════════════════════
# 资产缓存辅助函数
# ══════════════════════════════════════════════════════════

def _safe_filename(name: str) -> str:
    return _WIN_ILLEGAL.sub('_', name).strip()


def _asset_cache_dir(asset_folder: str) -> str:
    folder = (asset_folder or "").strip()
    if folder:
        return folder
    return os.path.join(folder_paths.get_output_directory(), "feicai_assets")


def _asset_cache_path(asset_folder: str, asset_type: str, name: str) -> str:
    safe = _safe_filename(name)
    if not safe:
        safe = "unnamed"
    filename = f"{asset_type}_{safe}.png"
    return os.path.join(_asset_cache_dir(asset_folder), filename)


def _load_asset_cache(path: str) -> object:
    try:
        if not os.path.isfile(path):
            return None
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        _log(f"[AssetCache] 命中: {os.path.basename(path)} ({img.size})")
        return pil2tensor(img)
    except Exception as e:
        _log(f"[AssetCache] 加载失败 {path}: {e}")
        return None


def _save_asset_cache(tensor: object, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pil = tensor2pil(tensor)
        pil.save(path, format="PNG")
        _log(f"[AssetCache] 保存: {os.path.basename(path)} ({pil.size})")
    except Exception as e:
        _log(f"[AssetCache] 保存失败 {path}: {e}")


# ══════════════════════════════════════════════════════════
# 故事板参考图智能筛选辅助函数
# ══════════════════════════════════════════════════════════

def _entity_in_prompt(prompt: str, name: str, alias_str: str) -> bool:
    names_to_check = [name]
    if alias_str:
        for a in alias_str.replace('，', ',').split(','):
            a = a.strip()
            if a and a not in names_to_check:
                names_to_check.append(a)
    for n in names_to_check:
        if n and n in prompt:
            return True
    return False


def _find_relevant_refs(prompt: str, parsed: dict, max_refs: int, ref_count: int) -> list:
    matched = []
    for i, char in enumerate(parsed.get('characters', [])):
        if _entity_in_prompt(prompt, char.get('name', ''), char.get('alias', '')):
            matched.append(i)
    scene_idx = len(parsed.get('characters', []))
    scene = parsed.get('scene')
    if scene and _entity_in_prompt(prompt, scene.get('name', ''), scene.get('alias', '')):
        matched.append(scene_idx)
    prop_start = scene_idx + (1 if scene else 0)
    for i, prop in enumerate(parsed.get('props', [])):
        if _entity_in_prompt(prompt, prop.get('name', ''), prop.get('alias', '')):
            matched.append(prop_start + i)
    if len(matched) > max_refs:
        matched = matched[:max_refs]
    if not matched:
        if scene:
            matched.append(scene_idx)
        if len(parsed.get('characters', [])) > 0:
            matched.append(0)
        matched = matched[:max_refs]
    matched = [idx for idx in matched if 0 <= idx < ref_count]
    return matched


def _describe_matched(selected_indices: list, parsed: dict) -> str:
    names = []
    chars = parsed.get('characters', [])
    scene = parsed.get('scene')
    props = parsed.get('props', [])
    scene_idx = len(chars)
    props_start = scene_idx + (1 if scene else 0)
    for idx in selected_indices:
        if idx < len(chars):
            names.append(chars[idx].get('name', f'角色#{idx}'))
        elif scene and idx == scene_idx:
            names.append(scene.get('name', f'场景#{idx}'))
        else:
            pi = idx - props_start
            if 0 <= pi < len(props):
                names.append(props[pi].get('name', f'道具#{idx}'))
            else:
                names.append(f'#{idx}')
    return ', '.join(names) if names else '(无)'


# ══════════════════════════════════════════════════════════
# GrsAI 通用请求/解析
# ══════════════════════════════════════════════════════════

def _size_to_aspect_ratio(size_str: str) -> str:
    """将 "WxH" 转为 "W:H" 格式，已是比例格式则原样返回。"""
    size_str = size_str.strip()
    if "x" in size_str or "X" in size_str:
        size_str = size_str.replace("x", ":").replace("X", ":")
    return size_str


def _grsai_generate(base_url: str, api_key: str, model: str, prompt: str,
                    images: list = None, aspect_ratio: str = "16:9",
                    timeout: int = 300, attempt: int = 0) -> torch.Tensor:
    """单次 /v1/api/generate 请求，成功返回 tensor，失败返回 None。"""
    body = {
        "model": model,
        "prompt": prompt,
        "images": images or [],
        "aspectRatio": aspect_ratio,
        "replyType": "json",
        "imageSize": "1K",
    }
    url = f"{base_url}/v1/api/generate"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    tag = f"(attempt={attempt})" if attempt > 0 else ""
    img_count = len(images or [])
    _log(f"  -> POST /v1/api/generate {tag} model={model} aspect={aspect_ratio} images={img_count}")
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    _log(f"  <- HTTP {resp.status_code} len={len(resp.content)} content_type={resp.headers.get('Content-Type','?')}")

    raw_text = resp.text.strip()
    if raw_text:
        _log(f"     raw[:300]={raw_text[:300]}")
    else:
        _log(f"     [空响应!] status={resp.status_code} url={url}")

    if not raw_text:
        return None

    if "data: " in raw_text:
        lines = raw_text.split("\n")
        json_candidates = []
        for line in lines:
            if line.startswith("data: "):
                candidate = line[6:].strip()
                if candidate and candidate != "[DONE]":
                    json_candidates.append(candidate)
        if json_candidates:
            all_results = []
            final_status = ""
            for candidate in json_candidates:
                try:
                    chunk = json.loads(candidate)
                    if chunk.get("status"):
                        final_status = chunk["status"]
                    if chunk.get("results"):
                        all_results.extend(chunk["results"])
                    elif chunk.get("url"):
                        all_results.append({"url": chunk["url"]})
                except json.JSONDecodeError:
                    _log(f"     SSE 行非 JSON: {candidate[:100]}")
                    continue
            if all_results or final_status:
                data = {"status": final_status or "succeeded", "results": all_results}
            else:
                try:
                    data = json.loads(json_candidates[-1])
                except json.JSONDecodeError:
                    _log(f"     SSE 解析失败: raw[:200]={raw_text[:200]}")
                    return None
        else:
            _log(f"     SSE 无有效 data 行: raw[:200]={raw_text[:200]}")
            return None
    else:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            _log(f"     JSON 解析失败: raw[:300]={raw_text[:300]}")
            return None

    if resp.status_code != 200:
        err = data.get('error', str(data))
        _log(f"  API错误: {err}")
        return None

    if data.get("status") == "succeeded" and data.get("results"):
        img_url = data["results"][0].get("url")
    elif data.get("url"):
        img_url = data["url"]
    else:
        _log(f"  响应无可用图片 URL: {str(data)[:200]}")
        return None

    if not img_url:
        return None

    return download_image(img_url, timeout=timeout)


# ══════════════════════════════════════════════════════════
# ImageGen — 图像生成 (generate / edit)
# ══════════════════════════════════════════════════════════

class ImageGen:
    """图像生成节点 — GrsAI /v1/api/generate"""

    QUALITIES = ["auto", "standard", "hd", "high", "medium", "low"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": GRSAI_API_KEY,
                    "placeholder": "sk-xxxxxxxx",
                }),
                "model": (MODEL_OPTIONS, {"default": "gpt-image-2"}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "提示词...",
                }),
                "image_mode": (["generate", "edit"], {"default": "generate"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
                "timeout": ("INT", {"default": 300, "min": 10, "max": 600}),
            },
            "optional": {
                "base_url": ("STRING", {
                    "default": GRSAI_BASE_URL,
                    "placeholder": "https://your-api.com/v1",
                }),
                "edit_image_1": ("IMAGE", {"tooltip": "[edit] 参考图 1"}),
                "edit_image_2": ("IMAGE", {"tooltip": "[edit] 参考图 2"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "raw_json")
    FUNCTION = "run"
    CATEGORY = "GrsAI"

    def run(self, api_key, model, prompt, image_mode, width, height, n, timeout,
            base_url=None, edit_image_1=None, edit_image_2=None):
        if not api_key.strip():
            return self._err("未填写 API Key")
        base_url = (base_url or GRSAI_BASE_URL).strip().rstrip("/")
        aspect = f"{width}:{height}"

        ref_images = None
        if image_mode == "edit":
            refs = [img for img in [edit_image_1, edit_image_2] if img is not None]
            if not refs:
                return self._err("[edit] 至少需要一张参考图")
            ref_pils = [tensor2pil(t[0] if len(t.shape) == 4 else t) for t in refs]
            ref_images = [_pil_to_base64_data_uri(p) for p in ref_pils]
            _log(f"edit 模式: {len(ref_images)} 张参考图")

        _log(f"mode={image_mode} model={model} aspect={aspect}")

        pbar = comfy.utils.ProgressBar(n + 1)
        pbar.update(1)

        tensors = []
        for i in range(n):
            t = _grsai_generate(base_url, api_key, model, prompt.strip(),
                                images=ref_images, aspect_ratio=aspect,
                                timeout=timeout)
            if t is not None:
                tensors.append(t)
            pbar.update(1)

        raw = json.dumps({"success": len(tensors), "total": n}, ensure_ascii=False)

        if not tensors:
            _log("全部生成失败!")
            return (placeholder(), raw)

        combined = torch.cat(tensors, dim=0)
        _log(f"完成: {len(tensors)}/{n} 张, combined shape={combined.shape}")
        return (combined, raw)

    def _err(self, msg):
        r = json.dumps({"error": msg}, ensure_ascii=False)
        _log(f"ERR: {msg}")
        return (placeholder(), r)


# ══════════════════════════════════════════════════════════
# AssetGen — 资产生成
# ══════════════════════════════════════════════════════════

class AssetGen:
    """脚本 -> 解析角色/场景/道具 -> 并行调 GrsAI /v1/api/generate -> 输出全部参考图"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": GRSAI_API_KEY,
                    "placeholder": "sk-xxxxxxxx",
                }),
                "script_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "粘贴完整脚本（角色/场景/道具/镜头）...",
                }),
                "model": (MODEL_OPTIONS, {"default": "gpt-image-2"}),
                "size": ("STRING", {
                    "default": "16:9",
                    "placeholder": "WxH 或 宽高比 如 16:9 / 1:1",
                    "tooltip": "输出宽高比，支持 WxH（如 1792x1024）或直接写比例（16:9）",
                }),
                "timeout": ("INT", {"default": 300, "min": 30, "max": 600}),
                "cache_assets": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用后按角色/场景/道具名缓存图片，同名不再重复生成",
                }),
                "asset_folder": ("STRING", {
                    "default": "",
                    "placeholder": "留空=output/feicai_assets；填写绝对路径自定义",
                    "tooltip": "资产缓存目录，图片以 {类型}_{名称}.png 命名",
                }),
            },
            "optional": {
                "base_url": ("STRING", {
                    "default": GRSAI_BASE_URL,
                    "placeholder": "https://your-api.com/v1",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "raw_json")
    FUNCTION = "run"
    CATEGORY = "GrsAI"

    def run(self, api_key, script_text, model, size, timeout,
            cache_assets=False, asset_folder="", base_url=None):
        if not api_key.strip():
            return self._err("未填写 API Key")
        if not script_text.strip():
            return self._err("script_text 为空")

        base_url = (base_url or GRSAI_BASE_URL).strip().rstrip("/")

        parsed = parse_script_text(script_text)
        _log(f"[AssetGen] 解析: {len(parsed['characters'])} 角色, "
             f"场景={'有' if parsed['scene'] else '无'}, "
             f"{len(parsed['props'])} 道具, {len(parsed['shots'])} 镜头")

        requests_list = []
        for char in parsed['characters']:
            requests_list.append({
                'name': char['name'], 'type': 'character',
                'prompt': char['description'], 'size': '16:9',
            })
        if parsed['scene']:
            requests_list.append({
                'name': parsed['scene']['name'], 'type': 'scene',
                'prompt': parsed['scene']['description'], 'size': '16:9',
            })
        for prop in parsed['props']:
            prop_prompt = (
                f"纯白色背景、产品三视图加特写大图设计。"
                f"画面左侧为同一物品的三视图，依次包含正面视图、侧面视图、背面视图，"
                f"三个视图等高等比例、完整入镜；画面右侧为同一物品的特写大图，细节清晰。"
                f"描述：{prop['description']}。"
                f"强调同一物品、白色背景、影视级真实质感、无裁切、无重叠。"
            )
            requests_list.append({
                'name': prop['name'], 'type': 'prop',
                'prompt': prop_prompt, 'size': '1:1',
            })

        total = len(requests_list)
        if total == 0:
            return self._err("未解析到任何角色/场景/道具")

        results = [None] * total
        pending_list = []

        if cache_assets:
            cache_root = _asset_cache_dir(asset_folder)
            _log(f"[AssetGen] 缓存模式: ON, 目录={cache_root}")
            for i, req in enumerate(requests_list):
                cache_path = _asset_cache_path(asset_folder, req['type'], req['name'])
                cached_tensor = _load_asset_cache(cache_path)
                if cached_tensor is not None:
                    results[i] = {'name': req['name'], 'tensor': cached_tensor,
                                  'error': None, 'cached': True}
                else:
                    pending_list.append((i, req))
        else:
            pending_list = list(enumerate(requests_list))

        pending_count = len(pending_list)
        cached_count = total - pending_count
        if cached_count > 0:
            _log(f"[AssetGen] 缓存命中 {cached_count}, 待生成 {pending_count}")
        else:
            _log(f"[AssetGen] 共 {total} 个请求, 并行执行...")

        pbar = comfy.utils.ProgressBar(total + 1)
        pbar.update(cached_count)

        def do_one(index, req):
            req_size = req.get('size', size)
            aspect = _size_to_aspect_ratio(req_size)
            for attempt in range(3):
                try:
                    t = _grsai_generate(base_url, api_key, model, req['prompt'],
                                        aspect_ratio=aspect, timeout=timeout,
                                        attempt=attempt)
                    if t is not None:
                        _log(f"  [{req['name']}] OK {t.shape}")
                        return (index, req['name'], t, None)
                except Exception as e:
                    _log(f"  [{req['name']}] 尝试{attempt + 1}/3 异常: {e}")
                    if attempt == 2:
                        import traceback
                        _log(f"  [{req['name']}] FAIL {traceback.format_exc()}")
            return (index, req['name'], None, "3次尝试均失败")

        if pending_count > 0:
            BATCH_SIZE = 3
            for start in range(0, pending_count, BATCH_SIZE):
                batch = pending_list[start:start + BATCH_SIZE]
                _log(f"[AssetGen] 批次 {start // BATCH_SIZE + 1}: {len(batch)} 个请求...")
                with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                    futures = [ex.submit(do_one, i, r) for i, r in batch]
                    for f in as_completed(futures):
                        i, name, tensor, err = f.result()
                        results[i] = {'name': name, 'tensor': tensor, 'error': err,
                                      'cached': False}
                        pbar.update(1)
                        if cache_assets and tensor is not None:
                            req = requests_list[i]
                            cache_path = _asset_cache_path(
                                asset_folder, req['type'], req['name'])
                            _save_asset_cache(tensor, cache_path)

        tensors = [r['tensor'] for r in results if r and r['tensor'] is not None]
        errors = [r for r in results if r and r['error']]

        raw = json.dumps({
            'total': total,
            'success': len(tensors),
            'failed': len(errors),
            'cached': sum(1 for r in results if r and r.get('cached')),
            'items': [{'name': r['name'], 'ok': r['tensor'] is not None,
                       'cached': r.get('cached', False),
                       'error': r['error']} for r in results if r],
        }, ensure_ascii=False, indent=2)

        if not tensors:
            _log("[AssetGen] 全部失败!")
            return (placeholder(), raw)

        tensors = _unify_tensor_sizes(tensors, "AssetGen")
        combined = torch.cat(tensors, dim=0)
        _log(f"[AssetGen] 完成: {len(tensors)}/{total} 成功, "
             f"缓存命中 {sum(1 for r in results if r and r.get('cached'))}, "
             f"combined shape={combined.shape}")
        pbar.update(1)
        return (combined, raw)

    def _err(self, msg):
        r = json.dumps({"error": msg}, ensure_ascii=False)
        _log(f"[AssetGen] ERR: {msg}")
        return (placeholder(), r)


# ══════════════════════════════════════════════════════════
# SceneCompositor — 场景合成（道具融入场景）
# ══════════════════════════════════════════════════════════

class SceneCompositor:
    """从 AssetGen 资产 batch 中取出场景和道具，将常驻道具 rembg 去底后
    通过 GrsAI /v1/api/generate (images 数组) 融入场景图。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": GRSAI_API_KEY,
                    "placeholder": "sk-xxxxxxxx",
                }),
                "script_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "粘贴完整脚本（与 AssetGen 使用同一脚本）...",
                }),
                "asset_images": ("IMAGE", {
                    "tooltip": "从 AssetGen.images 接过来的全部资产图 batch",
                }),
                "model": (MODEL_OPTIONS, {"default": "gpt-image-2"}),
                "size": ("STRING", {
                    "default": "16:9",
                    "placeholder": "宽高比 如 16:9 / 1:1",
                }),
                "timeout": ("INT", {"default": 300, "min": 30, "max": 600}),
            },
            "optional": {
                "base_url": ("STRING", {
                    "default": GRSAI_BASE_URL,
                    "placeholder": "https://your-api.com/v1",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("composite_scene", "raw_json")
    FUNCTION = "run"
    CATEGORY = "GrsAI"
    DESCRIPTION = "将场景描述中出现的常驻道具去底后融入场景图。场景未提及的道具不处理。"

    def run(self, api_key, script_text, asset_images, model, size, timeout,
            base_url=None):
        if not api_key.strip():
            return self._err("未填写 API Key")
        if not script_text.strip():
            return self._err("script_text 为空")
        if asset_images is None:
            return self._err("缺少 asset_images（请连接 AssetGen 输出）")

        base_url = (base_url or GRSAI_BASE_URL).strip().rstrip("/")
        aspect = _size_to_aspect_ratio(size)

        parsed = parse_script_text(script_text)
        scene = parsed.get('scene')
        props = parsed.get('props', [])
        chars = parsed.get('characters', [])

        if not scene:
            _log("[SceneCompositor] 脚本无场景，直通")
            return (asset_images, json.dumps({"skipped": "no scene"}, ensure_ascii=False))
        if not props:
            _log("[SceneCompositor] 脚本无道具，直通场景原图")
            scene_idx = len(chars)
            if scene_idx < asset_images.shape[0]:
                return (asset_images[scene_idx:scene_idx + 1],
                        json.dumps({"skipped": "no props"}, ensure_ascii=False))
            return self._err("batch 中无场景图")

        scene_desc = scene.get('description', '')
        scene_idx = len(chars)
        if scene_idx >= asset_images.shape[0]:
            return self._err(f"batch 索引越界: scene_idx={scene_idx}, batch={asset_images.shape[0]}")

        scene_tensor = asset_images[scene_idx:scene_idx + 1]

        resident_props = []
        skipped_props = []
        for i, prop in enumerate(props):
            if _entity_in_prompt(scene_desc, prop.get('name', ''), prop.get('alias', '')):
                resident_props.append((i, prop))
            else:
                skipped_props.append(prop.get('name', f'道具#{i}'))

        if not resident_props:
            _log(f"[SceneCompositor] 场景描述中未提及任何道具 "
                 f"(道具: {[p.get('name','?') for p in props]}), 直通场景原图")
            return (scene_tensor,
                    json.dumps({"skipped": "no resident props",
                                "all_props": [p['name'] for p in props]},
                               ensure_ascii=False))

        _log(f"[SceneCompositor] 常驻道具 {len(resident_props)} 个: "
             f"{[r[1]['name'] for r in resident_props]}; "
             f"跳过 {len(skipped_props)} 个: {skipped_props if skipped_props else '(无)'}")

        try:
            from rembg import remove
        except ImportError:
            _log("[SceneCompositor] rembg 未安装，道具不去底直接发送")
            remove = None

        scene_pil = tensor2pil(scene_tensor)
        image_b64s = [_pil_to_base64_data_uri(scene_pil)]

        prop_start_idx = scene_idx + 1
        prop_names = []
        for prop_i, prop in resident_props:
            batch_i = prop_start_idx + prop_i
            if batch_i >= asset_images.shape[0]:
                _log(f"  [SceneCompositor] 道具 {prop['name']} batch 索引 {batch_i} 越界, 跳过")
                continue

            prop_tensor = asset_images[batch_i:batch_i + 1]
            prop_pil = tensor2pil(prop_tensor)

            if remove is not None:
                try:
                    prop_pil = remove(prop_pil)
                    _log(f"  [SceneCompositor] {prop['name']} -> rembg 去底 OK")
                except Exception as e:
                    _log(f"  [SceneCompositor] {prop['name']} rembg 失败: {e}, 用原图")

            image_b64s.append(_pil_to_base64_data_uri(prop_pil))
            prop_names.append(prop['name'])

        if len(image_b64s) <= 1:
            _log("[SceneCompositor] 没有可用的道具图, 退回原场景")
            return (scene_tensor,
                    json.dumps({"error": "no valid prop images"}, ensure_ascii=False))

        prop_names_str = '、'.join(prop_names)
        edit_prompt = (
            f"{scene_desc}\n\n"
            f"【物品摆放指令】第一张参考图为场景底图。后续参考图为需要放置在场景中的物品"
            f"（已去底）：{prop_names_str}。请根据场景描述将这些物品自然放置在场景的合理位置，"
            f"保持场景原有的光照、色调和透视关系。物品的大小比例应与场景匹配。"
        )

        composite = None
        last_error = ""
        for attempt in range(3):
            try:
                tag = f"[SceneCompositor]" + (f" 重试{attempt}" if attempt > 0 else "")
                _log(f"  {tag} -> POST /v1/api/generate (底图场景 + {len(image_b64s) - 1} 道具)")
                t = _grsai_generate(base_url, api_key, model, edit_prompt,
                                    images=image_b64s, aspect_ratio=aspect,
                                    timeout=timeout, attempt=attempt)
                if t is not None:
                    composite = t
                    _log(f"  {tag} OK 合成场景 {composite.shape}")
                    break
                last_error = "API 返回无图片"
            except Exception as e:
                _log(f"  [SceneCompositor] 尝试{attempt + 1}/3 异常: {e}")
                if attempt == 2:
                    import traceback
                    _log(f"  [SceneCompositor] FAIL {traceback.format_exc()}")
                last_error = str(e)[:200]

        if composite is None:
            _log(f"[SceneCompositor] 合成失败, 退回原场景图. 原因: {last_error}")
            return (scene_tensor,
                    json.dumps({"error": "composite failed", "detail": last_error},
                               ensure_ascii=False))

        raw = json.dumps({
            "resident_props": prop_names,
            "skipped_props": skipped_props,
            "scene_name": scene.get('name', ''),
            "composite_shape": list(composite.shape),
        }, ensure_ascii=False, indent=2)
        _log(f"[SceneCompositor] 完成: 合成场景 {composite.shape}, "
             f"常驻道具 {len(prop_names)} 个已融入")
        return (composite, raw)

    def _err(self, msg):
        r = json.dumps({"error": msg}, ensure_ascii=False)
        _log(f"[SceneCompositor] ERR: {msg}")
        return (placeholder(), r)


# ══════════════════════════════════════════════════════════
# Storyboard — 故事板（参考图 + 镜头提示词 -> 并行生成）
# ══════════════════════════════════════════════════════════

class Storyboard:
    """参考图 + 镜头提示词 -> 并行调 GrsAI /v1/api/generate -> 输出故事板"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": GRSAI_API_KEY,
                    "placeholder": "sk-xxxxxxxx",
                }),
                "script_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "粘贴完整脚本（用于提取镜头图片提示词）...",
                }),
                "model": (MODEL_OPTIONS, {"default": "gpt-image-2"}),
                "size": ("STRING", {
                    "default": "16:9",
                    "placeholder": "宽高比 如 16:9 / 1:1",
                }),
                "timeout": ("INT", {"default": 300, "min": 30, "max": 600}),
                "max_refs": ("INT", {
                    "default": 4, "min": 1, "max": 99,
                    "tooltip": "每个镜头最多上传几张参考图",
                }),
            },
            "optional": {
                "base_url": ("STRING", {
                    "default": GRSAI_BASE_URL,
                    "placeholder": "https://your-api.com/v1",
                }),
                "reference_images": ("IMAGE", {
                    "tooltip": "从 AssetGen 接过来的参考图 batch",
                }),
                "composite_scene": ("IMAGE", {
                    "tooltip": "SceneCompositor 输出的合成场景图。连了则替换场景槽位、移除已合成道具",
                }),
                "failed_folder": ("STRING", {
                    "default": "",
                    "placeholder": "留空=output/feicai_failed；自定义绝对路径",
                    "tooltip": "违规失败镜头的黑图+提示词存档目录",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "raw_json", "success_ids")
    FUNCTION = "run"
    CATEGORY = "GrsAI"

    def run(self, api_key, script_text, model, size, timeout, max_refs=4,
            failed_folder="", base_url=None, reference_images=None, composite_scene=None):
        if not api_key.strip():
            return self._err("未填写 API Key")
        if not script_text.strip():
            return self._err("script_text 为空")
        if reference_images is None:
            return self._err("缺少 reference_images（请连接 AssetGen 输出）")

        base_url = (base_url or GRSAI_BASE_URL).strip().rstrip("/")
        aspect = _size_to_aspect_ratio(size)

        parsed = parse_script_text(script_text)
        shots = parsed.get('shots', [])
        if not shots:
            return self._err("未解析到镜头信息")

        _log(f"[Storyboard] 解析到 {len(shots)} 个镜头, "
             f"参考图 batch shape={reference_images.shape}")

        if composite_scene is not None:
            num_chars = len(parsed.get('characters', []))
            scene = parsed.get('scene')
            resident_indices = set()
            if scene:
                scene_desc = scene.get('description', '')
                for i, prop in enumerate(parsed.get('props', [])):
                    if _entity_in_prompt(scene_desc, prop.get('name', ''),
                                         prop.get('alias', '')):
                        resident_indices.add(i)
            resident_names = [parsed['props'][i]['name']
                              for i in resident_indices] if resident_indices else []

            prop_start = num_chars + 1
            target_h = reference_images.shape[1]
            target_w = reference_images.shape[2]
            if composite_scene.shape[1] != target_h or composite_scene.shape[2] != target_w:
                cs = composite_scene.permute(0, 3, 1, 2)
                cs = torch.nn.functional.interpolate(cs, size=(target_h, target_w),
                                                     mode='bilinear', align_corners=False)
                composite_scene = cs.permute(0, 2, 3, 1)
                _log(f"[Storyboard] 合成场景 resize 到 ({target_h}, {target_w})")

            new_parts = []
            for i in range(num_chars):
                new_parts.append(reference_images[i:i + 1])
            new_parts.append(composite_scene)
            for i in range(prop_start, reference_images.shape[0]):
                orig_prop_idx = i - prop_start
                if orig_prop_idx not in resident_indices:
                    new_parts.append(reference_images[i:i + 1])
            reference_images = torch.cat(new_parts, dim=0)
            if resident_indices:
                parsed['props'] = [p for idx, p in enumerate(parsed.get('props', []))
                                   if idx not in resident_indices]
                _log(f"[Storyboard] 合成场景已融入常驻道具: {resident_names}, "
                     f"剩余独立道具: {[p['name'] for p in parsed['props']]}, "
                     f"新 batch shape={reference_images.shape}")

        ref_count = reference_images.shape[0]
        ref_pils = []
        for i in range(ref_count):
            ref_pils.append(tensor2pil(reference_images[i:i + 1]))
        _log(f"[Storyboard] 参考图: {ref_count} 张")

        pbar = comfy.utils.ProgressBar(len(shots) + 1)
        pbar.update(0)
        results = [None] * len(shots)

        def do_one(index, shot):
            for attempt in range(3):
                try:
                    prompt = shot.get('图片提示词', '')
                    if not prompt:
                        return (index, shot['id'], None, "缺少图片提示词")

                    selected = _find_relevant_refs(prompt.strip(), parsed, max_refs, ref_count)
                    tag_matched = _describe_matched(selected, parsed)
                    _log(f"  [镜头{shot['id']}] 参考图: {len(selected)}/{ref_count} 张 -> {tag_matched}")

                    image_b64s = [_pil_to_base64_data_uri(ref_pils[ri]) for ri in selected]

                    tag = f"[镜头{shot['id']}]" + (f" 重试{attempt}" if attempt > 0 else "")
                    _log(f"  {tag} -> POST /v1/api/generate ({len(selected)} refs)")
                    t = _grsai_generate(base_url, api_key, model, prompt.strip(),
                                        images=image_b64s, aspect_ratio=aspect,
                                        timeout=timeout, attempt=attempt)
                    if t is not None:
                        _log(f"  {tag} OK {t.shape}")
                        return (index, shot['id'], t, None)

                except Exception as e:
                    _log(f"  [镜头{shot['id']}] 尝试{attempt + 1}/3 异常: {e}")
                    if attempt == 2:
                        import traceback
                        _log(f"  [镜头{shot['id']}] FAIL {traceback.format_exc()}")

            return (index, shot['id'], None, "3次尝试均失败")

        BATCH_SIZE = 3
        for start in range(0, len(shots), BATCH_SIZE):
            batch = list(enumerate(shots[start:start + BATCH_SIZE], start=start))
            _log(f"[Storyboard] 批次 {start // BATCH_SIZE + 1}: {len(batch)} 个镜头...")
            with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                futures = [ex.submit(do_one, i, s) for i, s in batch]
                for f in as_completed(futures):
                    i, sid, tensor, err = f.result()
                    results[i] = {'shotId': sid, 'tensor': tensor, 'error': err}
                    pbar.update(1)

        success_ids = [r['shotId'] for r in results if r and r['tensor'] is not None]
        fail_ids = [r['shotId'] for r in results if r and r['error']]
        errors = [r for r in results if r and r['error']]
        success_count = len(success_ids)

        try:
            w_s, h_s = aspect.split(':')
            bw, bh = int(w_s) * 64, int(h_s) * 64
        except Exception:
            bw, bh = 1920, 1080
        black_img = torch.zeros((1, bh, bw, 3), dtype=torch.float32)

        if fail_ids:
            failed_dir = (failed_folder or "").strip()
            if not failed_dir:
                failed_dir = os.path.join(
                    folder_paths.get_output_directory(), "feicai_failed")
            os.makedirs(failed_dir, exist_ok=True)
            shot_lookup = {s['id']: s for s in shots}
            for sid in fail_ids:
                safe_name = _safe_filename(f"镜头{sid}")
                pil_black = Image.new('RGB', (bw, bh), color=(0, 0, 0))
                pil_black.save(os.path.join(failed_dir, f"{safe_name}_黑图.png"))
                shot = shot_lookup.get(sid, {})
                prompt = shot.get('图片提示词', '') or ''
                txt_path = os.path.join(failed_dir, f"{safe_name}_prompt.txt")
                with open(txt_path, 'w', encoding='utf-8') as fh:
                    fh.write(prompt)
                _log(f"[Storyboard] 失败存档: 镜头{sid} -> {failed_dir}")

        tensors = []
        for r in results:
            if r and r['tensor'] is not None:
                tensors.append(r['tensor'])
            else:
                tensors.append(black_img)

        success_ids_json = json.dumps(success_ids, ensure_ascii=False)

        raw = json.dumps({
            'total': len(shots),
            'success': success_count,
            'failed': len(errors),
            'ref_count': ref_count,
            'success_ids': success_ids,
            'fail_ids': fail_ids,
            'shots': [{'shotId': r['shotId'], 'ok': r['tensor'] is not None,
                       'error': r['error']} for r in results if r],
        }, ensure_ascii=False, indent=2)

        if success_count == 0:
            _log(f"[Storyboard] 全部失败! 已存 {len(fail_ids)} 个黑图")
            return (placeholder(), raw, success_ids_json)

        tensors = _unify_tensor_sizes(tensors, "Storyboard")
        combined = torch.cat(tensors, dim=0)
        _log(f"[Storyboard] 完成: {success_count}/{len(shots)} 成功, "
             f"成功镜头ID: {success_ids}, combined shape={combined.shape}")
        pbar.update(1)
        return (combined, raw, success_ids_json)

    def _err(self, msg):
        r = json.dumps({"error": msg}, ensure_ascii=False)
        _log(f"[Storyboard] ERR: {msg}")
        return (placeholder(), r, "[]")


# ══════════════════════════════════════════════════════════
# RemoveBG — 去背
# ══════════════════════════════════════════════════════════

class RemoveBG:
    """使用 rembg 去除图片背景，输出 RGBA 或白底 RGB"""

    BG_MODES = ["white", "black", "transparent", "green"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "需要去背的图片"}),
                "background": (cls.BG_MODES, {"default": "white"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "run"
    CATEGORY = "GrsAI"

    def run(self, images, background):
        from rembg import remove

        batch = images.shape[0]
        pbar = comfy.utils.ProgressBar(batch)
        _log(f"[RemoveBG] 处理 {batch} 张图片, 背景={background}")

        out_images = []
        out_masks = []

        for i in range(batch):
            arr = images[i].cpu().numpy()
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)

            result = remove(pil)

            rgba = np.array(result).astype(np.float32) / 255.0
            rgb = rgba[:, :, :3]
            alpha = rgba[:, :, 3]

            mask = torch.from_numpy(alpha).unsqueeze(0)

            if background == "white":
                bg = np.ones_like(rgb)
            elif background == "black":
                bg = np.zeros_like(rgb)
            elif background == "green":
                bg = np.zeros_like(rgb)
                bg[:, :, 1] = 1.0
            else:
                bg = np.zeros_like(rgb)

            a3 = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            blended = rgb * a3 + bg * (1 - a3)

            out_images.append(torch.from_numpy(blended).unsqueeze(0))
            out_masks.append(mask)
            pbar.update(1)

        combined_img = torch.cat(out_images, dim=0)
        combined_mask = torch.cat(out_masks, dim=0)
        _log(f"[RemoveBG] 完成: {batch} 张, "
             f"image shape={combined_img.shape}, mask shape={combined_mask.shape}")
        return (combined_img, combined_mask)


# ══════════════════════════════════════════════════════════
# ScriptExtractor — 脚本提取器
# ══════════════════════════════════════════════════════════

class ScriptExtractor:
    """从脚本提取视频提示词，以 --- 分隔拼接，兼容 TextIndexSelect + for 循环。"""

    SEPARATOR = "\n\n---\n\n"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "粘贴完整脚本...",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("全部视频提示词", "视频提示词总数")
    FUNCTION = "run"
    CATEGORY = "GrsAI"
    DESCRIPTION = "从脚本提取每个镜头的视频提示词，以 --- 分隔符拼接"

    def run(self, script_text):
        if not script_text.strip():
            return ("", 0)
        parsed = parse_script_text(script_text)
        shots = parsed.get('shots', [])
        video_prompts = []
        for s in shots:
            vp = s.get('视频提示词', '')
            vp = vp.strip() if vp else ''
            if vp:
                video_prompts.append(vp)
        total = len(video_prompts)
        _log(f"[ScriptExtractor] 提取 {total} 个视频提示词")
        joined = self.SEPARATOR.join(video_prompts)
        return (joined, total)


# ══════════════════════════════════════════════════════════
# ShotFilter — 镜头过滤器
# ══════════════════════════════════════════════════════════

class ShotFilter:
    """接收 Storyboard 的 success_ids + ScriptExtractor 的提示词 -> 过滤仅输出有效镜头。"""

    SEPARATOR = "\n\n---\n\n"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "joined_prompts": ("STRING", {
                    "multiline": True, "default": "",
                    "placeholder": "从 ScriptExtractor 的 全部视频提示词 连过来",
                }),
                "prompt_count": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "ScriptExtractor 的 视频提示词总数（原始总数）",
                }),
                "success_ids": ("STRING", {
                    "default": "[]",
                    "placeholder": "[1, 2, 4]",
                    "tooltip": "Storyboard 的 success_ids（JSON 数组，1-based 镜头ID）",
                }),
            },
            "optional": {
                "storyboard_images": ("IMAGE", {
                    "tooltip": "Storyboard 输出的 images batch，连了则同步过滤黑图保持对齐",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "IMAGE")
    RETURN_NAMES = ("全部有效视频提示词", "有效视频提示词总数", "全部有效故事板")
    FUNCTION = "run"
    CATEGORY = "GrsAI"
    DESCRIPTION = "按 Storyboard 的 success_ids 过滤 ScriptExtractor 的提示词，可选同步过滤图片 batch"

    def run(self, joined_prompts, prompt_count, success_ids, storyboard_images=None):
        try:
            ids = json.loads(success_ids.strip() or "[]")
            if not isinstance(ids, list):
                ids = []
        except Exception:
            ids = []
        valid_set = set(ids)
        all_prompts = joined_prompts.split(self.SEPARATOR) if joined_prompts.strip() else []
        if not all_prompts and prompt_count > 0:
            all_prompts = [""] * prompt_count
        valid_prompts = []
        for i, p in enumerate(all_prompts):
            if (i + 1) in valid_set:
                valid_prompts.append(p)
        total = len(valid_prompts)
        _log(f"[ShotFilter] {prompt_count} 个镜头 -> {total} 个有效 "
             f"(success_ids={ids}, 已过滤 {prompt_count - total} 个失败镜头)")
        joined = self.SEPARATOR.join(valid_prompts)

        if storyboard_images is not None and len(valid_prompts) > 0:
            img_count = storyboard_images.shape[0]
            valid_imgs = []
            for shot_id in ids:
                idx = shot_id - 1
                if 0 <= idx < img_count:
                    valid_imgs.append(storyboard_images[idx])
            if valid_imgs:
                filtered_images = torch.cat([img.unsqueeze(0) for img in valid_imgs], dim=0)
            else:
                filtered_images = storyboard_images[:0]
            _log(f"[ShotFilter] 图片过滤: {img_count} -> {filtered_images.shape[0]} 张")
        elif storyboard_images is not None:
            filtered_images = storyboard_images[:0]
        else:
            filtered_images = torch.zeros((0, 64, 64, 3))

        return (joined, total, filtered_images)


# ══════════════════════════════════════════════════════════
# TTSExtractor — TTS 提示词提取器
# ══════════════════════════════════════════════════════════

class TTSExtractor:
    """从脚本提取每个镜头的 TTS 提示词，以 --- 分隔拼接，直连 IndexTTS2 text 端口。"""

    SEPARATOR = "\n\n---\n\n"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "粘贴完整脚本（含 TTS提示词 字段）...",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("全部TTS提示词", "TTS镜头总数")
    FUNCTION = "run"
    CATEGORY = "GrsAI"
    DESCRIPTION = "从脚本提取每个镜头的 TTS提示词，直接输出给 IndexTTS2 text 端口"

    def run(self, script_text):
        if not script_text.strip():
            return ("", 0)
        parsed = parse_script_text(script_text)
        shots = parsed.get('shots', [])
        tts_prompts = []
        for s in shots:
            tp = s.get('TTS提示词', '')
            tp = tp.strip() if tp else ''
            if tp and tp != '无':
                tts_prompts.append(tp)
        total = len(tts_prompts)
        _log(f"[TTSExtractor] 提取 {total} 个 TTS 提示词")
        joined = self.SEPARATOR.join(tts_prompts)
        return (joined, total)


# ══════════════════════════════════════════════════════════
# TTSFilter — TTS 提示词过滤器
# ══════════════════════════════════════════════════════════

class TTSFilter:
    """接收 Storyboard 的 success_ids + TTSExtractor 的 TTS 提示词 -> 过滤仅输出有效镜头的 TTS。"""

    SEPARATOR = "\n\n---\n\n"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "joined_tts": ("STRING", {
                    "multiline": True, "default": "",
                    "placeholder": "从 TTSExtractor 的 全部TTS提示词 连过来",
                }),
                "tts_count": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "TTSExtractor 的 TTS镜头总数",
                }),
                "success_ids": ("STRING", {
                    "default": "[]",
                    "placeholder": "[1, 2, 4]",
                    "tooltip": "Storyboard 的 success_ids（JSON 数组，1-based 镜头ID）",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("全部有效TTS提示词", "有效TTS镜头数")
    FUNCTION = "run"
    CATEGORY = "GrsAI"
    DESCRIPTION = "按 Storyboard 的 success_ids 过滤 TTSExtractor 的 TTS 提示词"

    def run(self, joined_tts, tts_count, success_ids):
        try:
            ids = json.loads(success_ids.strip() or "[]")
            if not isinstance(ids, list):
                ids = []
        except Exception:
            ids = []
        valid_set = set(ids)
        all_tts = joined_tts.split(self.SEPARATOR) if joined_tts.strip() else []
        if not all_tts and tts_count > 0:
            all_tts = [""] * tts_count
        valid_tts = []
        for i, p in enumerate(all_tts):
            if (i + 1) in valid_set:
                valid_tts.append(p)
        total = len(valid_tts)
        _log(f"[TTSFilter] {tts_count} 个 TTS -> {total} 个有效 "
             f"(success_ids={ids}, 已过滤 {tts_count - total} 个失败镜头)")
        joined = self.SEPARATOR.join(valid_tts)
        return (joined, total)


# ══════════════════════════════════════════════════════════
# AudioConcat — 音频拼接
# ══════════════════════════════════════════════════════════

class AudioConcat:
    """for 循环音频累积专用。任意一端为空时直接返回另一端。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio1": ("AUDIO",),
            "audio2": ("AUDIO",),
            "direction": (["right", "left"], {"default": "right"}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "concat"
    CATEGORY = "GrsAI"
    DESCRIPTION = "拼接两段音频。任意一端为 None 时返回另一端。"

    def concat(self, audio1, audio2, direction):
        if audio1 is None and audio2 is None:
            _log("[AudioConcat] 两端均为 None, 返回 None")
            return (None,)
        if audio1 is None:
            return (audio2,)
        if audio2 is None:
            return (audio1,)
        sample_rate_1 = audio1["sample_rate"]
        sample_rate_2 = audio2["sample_rate"]
        if sample_rate_1 != sample_rate_2:
            raise ValueError(
                f"采样率不一致: audio1={sample_rate_1}, audio2={sample_rate_2}")
        waveform_1 = audio1["waveform"]
        waveform_2 = audio2["waveform"]
        if direction == "right":
            concatenated = torch.cat((waveform_1, waveform_2), dim=2)
        elif direction == "left":
            concatenated = torch.cat((waveform_2, waveform_1), dim=2)
        return ({"waveform": concatenated, "sample_rate": sample_rate_1},)


# ══════════════════════════════════════════════════════════
# 节点注册
# ══════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "ImageGen": ImageGen,
    "AssetGen": AssetGen,
    "SceneCompositor": SceneCompositor,
    "Storyboard": Storyboard,
    "ShotFilter": ShotFilter,
    "RemoveBG": RemoveBG,
    "ScriptExtractor": ScriptExtractor,
    "TTSExtractor": TTSExtractor,
    "TTSFilter": TTSFilter,
    "Nienai-AudioConcat": AudioConcat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageGen": "ImageGen (图像生成)",
    "AssetGen": "AssetGen (资产生成)",
    "SceneCompositor": "SceneCompositor (场景合成)",
    "Storyboard": "Storyboard (故事板)",
    "ShotFilter": "ShotFilter (镜头过滤器)",
    "RemoveBG": "RemoveBG (去背)",
    "ScriptExtractor": "ScriptExtractor (脚本提取器)",
    "TTSExtractor": "TTSExtractor (TTS提取器)",
    "TTSFilter": "TTSFilter (TTS过滤器)",
    "Nienai-AudioConcat": "Nienai-AudioConcat (音频拼接)",
}
