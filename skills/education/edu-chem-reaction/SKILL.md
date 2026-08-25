---
name: edu-chem-reaction
description: Interactive chemistry reaction lessons driven by SymPy.
version: 1.0.0
author: WY (@akokoi1) + Hermes Agent
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [education, chemistry, reaction, sympy, interactive-lesson]
    category: education
    related_skills: []
    upstream: https://github.com/wy51ai/edulab
    upstream_commit: cf0bc1d68b4ea64307f57d7fac64667e6a3148cc
---

# Edu Chemistry Reaction Skill

Use this skill for exact, interactive chemistry-reaction lessons. It does not
upload the reference template unchanged; the final Edu artifact must be a self-contained 2D HTML payload.

## When to Use
一个可直接用浏览器打开的单页 HTML：一侧是反应对应的 **3D 分子动画**（Three.js，可旋转缩放，
拖动“反应进度”滑块逐帧看 **化学键断裂/生成、原子重新组合**，分步高亮 + 浮动分子标签），
另一侧是 **KaTeX 反应方程 + 分步讲解 + 原子守恒计数器**，并可选 **能量-反应进程曲线**、
火焰、催化剂质子、电子转移等叠加层。形态与 `templates/reaction.html` 一致。

## Prerequisites

This skill is available only in the Edu app scope. Use `skill_view` to load the
linked references, templates, and scripts when needed; do not use this skill in
Canvas-scoped requests or call Canvas-only tools.

The current Edu artifact uploader is the `generate_game` / `canvas_generate_game`
tool and accepts only a self-contained 2D HTML artifact with inline code/assets
and its required 1920x1080 stage contract. Do not submit this upstream Three.js
template unchanged: it uses external CDN dependencies and is a lesson reference,
not a valid upload payload. Preserve the balanced equation, atom-map checks, and
reaction data when producing an artifact that satisfies the current uploader
contract. RDKit remains optional and must not be installed automatically.

### Runtime dependencies

在 Edu agent 中通过 `execute_code` 运行 reaction kernel；该工具使用 Hermes 沙箱并已预装
`sympy`。只输出配平结果、原子映射、键变化和其他结构化 reaction data；不要依赖沙箱中的
HTML 文件，因为执行结束后工作区会被清理。模型根据返回的数据组装符合 uploader contract
的自包含 2D HTML，并把 HTML 直接传给 `generate_game` 的 `html` 字段；不要把 Python 文件
或临时截图作为交付物。
计算核心 `lib/reaction_kernel.py` 依赖 **sympy**。Edu 的隔离运行时镜像必须预装
sympy 1.14；若执行预检报告缺失，返回部署配置错误，不要在请求期间安装依赖或切换到宿主机解释器。
**RDKit 是可选项**：装了则混合几何会用它由 SMILES 生成真实构象，没装就用自建 VSEPR 库——
两种都能跑，**本技能任何时候都不会自动安装 RDKit**。

## How to Run

Run the bundled kernel through `execute_code`, return only structured reaction data, then pass a self-contained 2D HTML string to `generate_game`.

## Quick Reference

- `lib/reaction_kernel.py`: balancing, conservation, atom mapping, and bond changes.
- `templates/reaction.html`: reference renderer and data schema.
- `scripts/generate.py`: local registry and deterministic examples.

## Procedure

### 第 1 步：得到 reaction spec（三入口归一）
把反应整理成结构化 spec（格式见 `references/problem-schema.md`）：反应物/产物、原子映射或显式原子、
条件（点燃/通电/催化/可逆）、所属类别、分步讲解、**语言**。
- **文字反应/方程**：直接抽取反应物与产物，调 kernel 自动配平。
- **图片**：用视觉读图抽取方程，并**把识别到的反应回显给用户确认**（方程/条件/类别/语言）后再继续。
- **随机出题**：从注册表挑一个反应，或在库内物种间组合并用 `balanced_coefficients` 配平、答案规整再用。

> **输出语言跟随提示词语言**：英文提示 → 英文网页，中文 → 中文。spec 里记下 `meta.language`。

### 第 2 步：用 kernel 精确计算（不要心算）
按 `references/conventions.md` 的建模约定，调用 `lib/reaction_kernel.py`：
- `balanced_coefficients(...)` 用 sympy 零空间**自动配平**（方程系数有保证）；
- `assemble_data(spec)` 展开分子实例、**校验原子守恒与原子映射双射**、**推导键的断/成（差集）**、
  算出每个原子在反应物态/产物态的世界坐标，产出注入模板的 `data`。

`execute_code` 接收 Python 源码，不接收 shell 命令。内核目录已经加入
`PYTHONPATH`，所以直接导入模块并打印结构化结果：
```python
import json
import reaction_kernel as K

# 将这里的 spec 替换为用户题目中的 reaction spec。
data = K.assemble_data(spec)
print(json.dumps(data, ensure_ascii=False))
```

### 第 3 步：写 build_* 拼 spec 并注入模板

> 📍 **唯一产物（重要）**：成品 HTML 通过 `generate_game` 的 `html` 字段上传。
> `execute_code` 的工作区是临时的，参考 HTML、脚本和截图都必须在执行结束时清理，
> 不要把它们当作交付文件。

照着 `scripts/generate.py` 里的 `build_*` 提取结构化 reaction data 并从 `execute_code`
输出；模型根据返回的数据组装符合 uploader contract 的自包含 2D HTML，并直接上传：
```python
import json
print(json.dumps(K.assemble_data(spec), ensure_ascii=False))
```
**范例（直接照抄改）**：
- `build_combustion_ch4`（甲烷燃烧·morph·火焰·能量）——高层 `species + atom_map` 的范本；
- `build_redox_na_cl2`（钠+氯气·氧化还原·电子转移）——叠加 `electrons`；
- `build_esterification`（酯化·mechanism·催化剂·过渡态）——低层 `atoms + fragments + 关键帧` 的范本。

`scripts/generate.py` 是参考实现；在 agent 中不要把它当作 shell 命令执行。
如需复用其中的构造逻辑，把所需的 Python 函数复制为 `execute_code` 源码，
导入 `reaction_kernel` 和 `molecules`，并只打印结构化数据。

## Pitfalls

- Do not use a host interpreter or install packages during a request; the isolated runtime is immutable and network-disabled.
- Do not upload the reference template, Python files, screenshots, or temporary sandbox paths as the final artifact.

## Verification
- sympy 配平系数 == 方程展示系数 == 各分子实例个数（`assemble_data` 内已断言）。
- 原子映射是反应物↔产物原子的**双射**、元素一致；键端点都存在（kernel 已校验）。
- 原子守恒计数器在反应前后不变（催化剂不计入）。
- Use `execute_code` for deterministic kernel assertions and structural checks on the generated HTML:
  confirm the reaction data parses, mapped atoms and bond endpoints are present, and no external
  runtime dependency or temporary file is required.
- Submit the self-contained HTML with `generate_game` and verify the returned artifact result. Do not
  create a local browser server or persist screenshots in the sandbox.

### Delivery
成品通过 `generate_game` 上传并把返回的媒体结果交付给用户。交付前确认没有遗留沙箱临时文件，
也不要把沙箱中的临时文件路径告诉用户。

## 两套引擎与自动选择
模板内置一套统一渲染器、两种逐帧定位（共用键差绘制/标签/叠加层/UI）：
- **morph**（原子变形）：原子各自从反应物态插值到产物态，天然展示**原子守恒/重组**，适配任意反应。
- **mechanism**（机理关键帧）：原子归属刚体片段（fragment），按 K0/K1/K2 关键帧整体位移，
  基团不变形，适配**催化剂/过渡态/离去基团**类有机机理。

`assemble_data` 据 `meta.engine`（`auto`/`morph`/`mechanism`）选择：`auto` 时，类别为 `organic`
或带 `fragments` 走 mechanism，否则走 morph。叠加层均为数据开关（见 schema）：
`flame`（燃烧/强放热）、`catalyst`（催化剂质子+开关）、`transitionGlow`（过渡态能量光）、
`electrons`（氧化还原电子转移）、`energy`（能量-反应进程曲线）、原子守恒计数器（默认开）。

**配色**：整体为亮色（教科书球棍图风：原子带深色描边 + 柔和投影 + 白底面板），各反应用 `meta.accent`
区分强调色（燃烧 amber、酯化 indigo、钠氯 violet…）。

## 扩展
- **加反应**：在 `generate.py` 加一个 `build_*`（高层 `species+atom_map`，或低层 `atoms+fragments`），
  注册进 `REGISTRY`。
- **加分子/离子**：在 `lib/molecules.py` 的 `_LIBRARY_BUILDERS` 加一项（VSEPR 几何 + 显示元数据 + 内部键）。
- **加叠加层**：在 `templates/reaction.html` 增一个由 `data` 字段驱动的可选模块。

## 目录
- `templates/reaction.html` — 数据驱动模板（统一渲染器 + 双引擎 + 数据岛 `__REACTION_DATA__`）
- `lib/molecules.py` — VSEPR 理想分子几何库（含元素表/配色/半径）
- `lib/reaction_kernel.py` — sympy 配平 + 守恒/映射校验 + 键差 + 场景装配 + 可选 RDKit 探测
- `scripts/generate.py` — 注入模板 + 范例 build_*（含 REGISTRY 与 CLI）
- `references/problem-schema.md` — reaction spec 与 data 的数据格式
- `references/conventions.md` — 建模约定、引擎选择、叠加层、配平与自检
