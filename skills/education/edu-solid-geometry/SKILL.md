---
name: edu-solid-geometry
description: Interactive solid geometry lessons driven by SymPy.
version: 1.0.0
author: WY (@akokoi1) + Hermes Agent
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [education, geometry, solid-geometry, sympy, interactive-lesson]
    category: education
    related_skills: []
    upstream: https://github.com/wy51ai/edulab
    upstream_commit: cf0bc1d68b4ea64307f57d7fac64667e6a3148cc
---

# Edu Solid Geometry Skill

Use this skill for exact, interactive solid-geometry lessons. It does not
upload the reference template unchanged; the final Edu artifact must be a self-contained 2D HTML payload.

## When to Use
一个可直接用浏览器打开的单页 HTML：左侧题面/答案/分步解析（公式用 MathJax），
右侧是题目对应的 3D 模型（Three.js，可旋转缩放，分步高亮关键元素并切换镜头）。
形态与 `templates/lesson.html` 一致。

## Prerequisites

This skill is available only in the Edu app scope. Use `skill_view` to load the
linked references, templates, and scripts when needed; do not use this skill in
Canvas-scoped requests or call Canvas-only tools.

The current Edu artifact uploader is the `generate_game` / `canvas_generate_game`
tool and accepts only a self-contained 2D HTML artifact with inline code/assets
and its required 1920x1080 stage contract. Do not submit this upstream Three.js
template unchanged: it uses external CDN dependencies and is a lesson reference,
not a valid upload payload. Preserve the kernel-derived facts and lesson structure
when producing an artifact that satisfies the current uploader contract.

### Runtime dependencies

在 Edu agent 中通过 `execute_code` 运行 bundled kernels；该工具使用 Hermes 沙箱并已预装
`sympy`。只输出精确坐标、步骤和 lesson data；不要依赖沙箱中的 HTML 文件，因为执行结束后
工作区会被清理。模型根据返回的数据组装符合 uploader contract 的自包含 2D HTML，并把 HTML
直接传给 `generate_game` 的 `html` 字段；不要把 Python 文件或临时截图作为交付物。
计算核心 `lib/geometry_kernel.py` 依赖 **sympy**。Edu 的隔离运行时镜像必须预装
sympy 1.14；若执行预检报告缺失，返回部署配置错误，不要在请求期间安装依赖或切换到宿主机解释器。

## How to Run

Run the bundled kernel through `execute_code`, return only structured lesson data, then pass a self-contained 2D HTML string to `generate_game`.

## Quick Reference

- `lib/geometry_kernel.py`: exact coordinates and geometry calculations.
- `templates/lesson.html`: reference renderer and data schema.
- `scripts/generate.py`: local registry and deterministic examples.

## Procedure

### 第 1 步：得到 problem spec（三入口归一）
把题目整理成结构化 spec（格式见 `references/problem-schema.md`）：几何体类型与尺寸、
已知构造点/条件、所求类型与对象、**语言**。
- **文字题目**：直接抽取。
- **图片**：用视觉读图抽取，并**把识别到的题目回显给用户确认**（题面/几何体/尺寸/所求/语言）后再继续。
- **随机出题**：选定几何体与题型，用 kernel 随机参数求解，答案不规整就重抽。

> **输出语言跟随提示词语言**：英文提示 → 英文网页，中文 → 中文。spec 里记下 `language`。

### 第 2 步：用 kernel 精确计算（不要心算）
按 `references/conventions.md` 的建系约定与解法配方，调用 `lib/geometry_kernel.py`：
得到精确坐标、关键向量、法向量、最终答案，以及各步骤要展示的中间量（均为 LaTeX 字符串）。
顶点的 three.js 坐标用 `kernel.to_three(points, scale)` 得到。

`execute_code` 接收 Python 源码，不接收 shell 命令。内核目录已经加入
`PYTHONPATH`，所以直接导入模块并打印结构化结果：
```python
import json
import geometry_kernel as K

data = K.solve_cube_line_plane_angle(edge=1, scale=2)
print(json.dumps(data, ensure_ascii=False, default=str))
```

### 第 3 步：组装 lesson data 并注入模板

> 📍 **唯一产物（重要）**：成品 HTML 通过 `generate_game` 的 `html` 字段上传。
> `execute_code` 的工作区是临时的，参考 HTML、脚本和截图都必须在执行结束时清理，
> 不要把它们当作交付文件。

写一个**临时构建脚本**，导入 kernel、bodies，拼出 `lesson` / `steps` / `model` 数据
（schema 见 `references/problem-schema.md`），并从 `execute_code` 输出 JSON。模型根据返回的数据
组装符合 uploader contract 的自包含 2D HTML，再直接传给 `generate_game`：

```python
import json
print(json.dumps(data, ensure_ascii=False))
```

- `steps[*].content` 里的所有数值**直接引用 kernel 的计算结果**，模型只负责组织讲解文字（按目标语言书写）。
- `model.points` 用 `kernel.to_three(...)` 的结果；`model.spheres`/`edges` 用 `lib/bodies.py` 的拓扑
  （`quad_pyramid` / `tri_pyramid` / `cuboid` / `cube` / `prism`），罕见几何体可手写 edges。
- 每步配 `highlight`（该步可见元素的绝对集合）与 `cameraPos`。
- **题面给出线段长度时**：为对应棱加 `measure` 元素（`label` 用 LaTeX，如 `2\sqrt{2}`），
  并把它放进"建系/列已知条件"那步的 `highlight`，在 3D 图中点处标出长度（见 problem-schema）。
- 英文输出时填 `lesson.ui` 英文文案并设 `lesson.language="en"`。

**可直接参考的范例**：`scripts/generate.py` 里的 `build_data()`（正四棱锥·线面角）、
`build_cube_data()`（正方体·线面角）、`build_box_volume_data()`（长方体·体积）都是完整范本，照着改即可。

`scripts/generate.py` 是参考实现；在 agent 中不要把它当作 shell 命令执行。
如需复用其中的构造逻辑，把所需的 Python 函数复制为 `execute_code` 源码，
导入 `geometry_kernel` 和 `bodies`，并只打印结构化数据。随机出题时在同一段
Python 中生成参数，用 `geometry_kernel.is_clean(...)` 过滤答案，再输出 JSON。
扩展随机题型时沿用"随机参数 → 求解 → is_clean 不过就重抽"。

## Pitfalls

- Do not use a host interpreter or install packages during a request; the isolated runtime is immutable and network-disabled.
- Do not upload the reference template, Python files, screenshots, or temporary sandbox paths as the final artifact.

## Verification
- kernel 答案 == 答案卡 `answerValue` == 末步骤展示的最终值（generate.py 已有断言示例）。
- 3D 顶点坐标来自 `kernel.to_three`（与解题同源）。
- Use `execute_code` for deterministic kernel assertions and structural checks on the generated HTML:
  confirm the lesson data parses, all highlighted model elements have matching IDs, and no external
  runtime dependency or temporary file is required.
- Submit the self-contained HTML with `generate_game` and verify the returned artifact result. Do not
  create a local browser server or persist screenshots in the sandbox.

### Delivery
成品通过 `generate_game` 上传并把返回的媒体结果交付给用户。交付前确认没有遗留沙箱临时文件，
也不要把沙箱中的临时文件路径告诉用户。

## 扩展
- **加题型**：在 `geometry_kernel.py` 加求解函数（见 conventions 配方表），在 `generate.py` 加一个 `build_*`。
- **加几何体**：在 `geometry_kernel.py` 加坐标构建函数，在 `bodies.py` 加棱拓扑。

## 目录
- `templates/lesson.html` — 数据驱动模板（通用 3D 渲染器 + 数据岛 `__LESSON_DATA__`）
- `lib/geometry_kernel.py` — sympy 精确计算核心
- `lib/bodies.py` — 几何体棱拓扑库
- `scripts/generate.py` — 注入模板 + 范例构建函数
- `references/problem-schema.md` — 数据格式
- `references/conventions.md` — 建系约定、解法配方、自检
