# Nienai-Nodes

Nienai 自用的 ComfyUI 节点包：**GrsAI 云端图像生成** + 短剧制作辅助节点（剧本解析 / 故事板 / 音频拼接）。
界面中文，分类统一在 `GrsAI` 下。

## 节点列表

| 节点 id | 显示名 | 作用 |
|---|---|---|
| `ImageGen` | ImageGen (图像生成) | 调 GrsAI 接口文生图 / 图生图 |
| `AssetGen` | AssetGen (资产生成) | 按资产档案批量生成角色 / 道具 / 场景参考图 |
| `SceneCompositor` | SceneCompositor (场景合成) | 把角色、道具合成到场景里出图 |
| `Storyboard` | Storyboard (故事板) | 生成/拼装多宫格故事板 |
| `ShotFilter` | ShotFilter (镜头过滤器) | 按镜头索引筛选图像批次 |
| `RemoveBG` | RemoveBG (去背) | 抠背景 |
| `ScriptExtractor` | ScriptExtractor (脚本提取器) | 从剧本/分镜 JSON 里提取字段 |
| `TTSExtractor` | TTSExtractor (TTS提取器) | 提取配音文本与音色档案 |
| `TTSFilter` | TTSFilter (TTS过滤器) | 过滤 / 整理 TTS 输出 |
| `Nienai-AudioConcat` | Nienai-AudioConcat (音频拼接) | 把多段音频按顺序拼成一条 |

## 安装

1. 把本目录放进 `ComfyUI/custom_nodes/Nienai-Nodes`；
2. 安装依赖：`pip install requests pillow numpy`（torch / numpy 一般 ComfyUI 自带）；
3. 配置 GrsAI 的 API Key（**不要写进源码**，二选一）：
   - 环境变量：`set GRSAI_API_KEY=sk-xxxx`（Windows）；
   - 或在本目录放一个 `.grsai_key` 文件，内容就是一行 key（该文件已在 `.gitignore` 里，不会被提交）。
4. 重启 ComfyUI。

> 节点上的 `api_key` 输入框默认值会自动取上面配置的 key；也可以临时在节点上覆盖填写。

## 目录

```
Nienai-Nodes/
├── __init__.py
├── nodes.py     所有节点的实现（约 1400 行）
├── pyproject.toml
└── README.md
```

## 说明

- 所有节点分类为 `GrsAI`，方便在节点搜索里一次性找到；
- 图像接口走 `https://grsai.dakka.com.cn`，支持 `gpt-image-2` / `nano-banana-2` 等模型；
- 与配套的 [ComfyUI-Nienai-easy](https://github.com/) 视频插件配合使用，可形成「剧本 → 图 → 视频」的流水线。
