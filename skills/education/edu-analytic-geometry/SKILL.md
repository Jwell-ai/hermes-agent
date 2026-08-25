---
name: edu-analytic-geometry
description: Interactive analytic geometry lessons driven by SymPy.
version: 1.0.0
author: WY (@akokoi1) + Hermes Agent
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [education, geometry, conic-sections, sympy, interactive-lesson]
    category: education
    related_skills: []
    upstream: https://github.com/wy51ai/edulab
    upstream_commit: cf0bc1d68b4ea64307f57d7fac64667e6a3148cc
---

# Edu Analytic Geometry Skill

Use this skill for exact, interactive analytic-geometry lessons. It does not
upload the reference template unchanged; the final Edu artifact must be a self-contained 2D HTML payload.

## When to Use
一个可直接用浏览器打开的单页 HTML（三栏）：
- **左栏**：题面 + 动态控制台 —— 一个可变参数滑块（如直线倾斜角 θ / 动点参数 t）驱动实时
  重算的几何量（交点坐标、斜率、数量积、弦长、面积…），以及"理论范围条"或"定值指示"。
- **中栏**：分步解析（公式用 **KaTeX**），可一键收起把空间让给画板。
- **右栏**：2D Canvas 动态几何画板（圆锥曲线 + 动直线/动点 + 向量 + 点标注 + 网格坐标轴），
  叠加画笔涂鸦工具栏。

形态与本技能的 `templates/board.html` 一致。

## Prerequisites

This skill is available only in the Edu app scope. Use `skill_view` to load the
linked references, templates, and scripts when needed; do not use this skill in
Canvas-scoped requests or call Canvas-only tools.

The current Edu artifact uploader is the `generate_game` / `canvas_generate_game`
tool and accepts only a self-contained 2D HTML artifact with inline code/assets
and its required 1920x1080 stage contract. Do not submit this upstream template
unchanged: it uses external CDN dependencies and is a lesson reference, not a
valid upload payload. Preserve the kernel-derived facts and interactive board
logic when producing an artifact that satisfies the current uploader contract.

### Runtime dependencies
计算核心 `lib/analytic_kernel.py` 依赖 **sympy**。Edu 的隔离运行时镜像必须预装
sympy 1.14；若执行预检报告缺失，返回部署配置错误，不要在请求期间安装依赖或切换到宿主机解释器。

## How to Run

Run the bundled kernel through `execute_code`, return only structured kernel data, then pass a self-contained 2D HTML string to `generate_game`.

## Quick Reference

- `lib/analytic_kernel.py`: exact geometry calculations.
- `templates/board.html`: reference renderer and data schema.
- `scripts/generate.py`: local registry and deterministic examples.

## Procedure

### 第 1 步：得到 problem spec（三入口归一）
把题目整理成结构化 spec（曲线类型与参数、已知点/条件、所求类型与对象、语言）。
- **文字题**：直接抽取。
- **图片**：视觉读图抽取，并**把识别到的题目回显给用户确认**（题面/曲线/参数/所求/语言）再继续。
- **随机出题**：选曲线 + 题型，随机参数 → kernel 求解，用 `analytic_kernel.is_clean(...)` 判答案
  是否规整，不规整就重抽。

> **输出语言跟随提示词语言**：英文提示 → 英文网页，中文 → 中文。spec 记下 `language`。

### 第 2 步：用 kernel 精确计算（不要心算）

在 Edu agent 中通过 `execute_code` 运行这些 Python kernels；该工具使用 Hermes
沙箱并已预装 `sympy`。只从脚本输出精确的 kernel 数据、步骤和交互参数；不要依赖
`Path.cwd()` 中的文件，因为沙箱任务结束后工作区会被清理。模型读取返回的数据后，
再生成一个符合上方 uploader contract 的自包含内联 CSS/JS 2D HTML，并把 HTML 直接传给
`generate_game` 的 `html` 字段。不要把计算过程改写成模型心算，也不要把 Python 文件当作最终产物。
按 `references/conventions.md` 的解法配方，调用 `lib/analytic_kernel.py` 与 `lib/conics.py`：
- `conics.ellipse/hyperbola/parabola/circle(...)` 得曲线对象（精确 a,b,c、焦点、顶点、准线、
  渐近线、`eq_latex`、以及给前端引擎的 `board` dict）。
- `chord_setup(conic, through)` 联立含参直线 `x=my+c` 得 y 的二次方程 + 韦达量（精确）。
- 目标量：`dot_product_expr` / `chord_len_sq_expr` / `triangle_area_expr` / `slope_product_central` …
- 取值范围：`range_over_m(expr, horizontal_valid=?)` —— **含开闭端点判定**（关键正确性点，见下）。
- 定值：`is_constant_in_m(expr)`。

`execute_code` 接收 Python 源码，不接收 shell 命令。内核目录已经加入
`PYTHONPATH`，所以直接导入模块并打印结构化结果：
```python
import json
import analytic_kernel as K
import conics

conic = conics.ellipse(5, 3)
expr, setup = K.dot_product_expr(conic, (0, 0), (1, 0))
print(json.dumps({"expr": K.tex(expr), "discriminant": K.tex(setup["disc"])}, ensure_ascii=False))
```

> ⚠️ **端点开闭 = 正确性命门**：过焦点的弦，水平线（x 轴，θ=0）与竖直线（θ=90）都是合法直线，
> 它们取到的端点要计入。例：椭圆 MA·MB 题，x 轴取到 −3、竖直线取到 7/4，故答案是**闭区间**
> `[-3, 7/4]`（很多教辅误写成开的 `(-3, 7/4]`）。`range_over_m` 已据此判定，且这样答案与交互
> 工具一致——拖滑块到 0° 就读到 −3。抛物线焦点弦的"轴方向"是退化线（只交一点），其极限端点
> 不计入（`horizontal_valid=False` 或限制 param 范围）。

### 第 3 步：组装数据并注入模板

> 📍 **唯一产物（最重要）**：交付给用户的**只有一个 `.html`**，通过 `generate_game` 的
> `html` 字段上传。`execute_code` 的工作区是临时的，构建脚本（`.py`）、`__pycache__`、
> 自检截图（`.png`）和参考 HTML 都必须留在沙箱内并在执行结束时清理；不要把它们作为交付物。

把"组装数据"写成传给 `execute_code` 的 Python 源码，让它输出 JSON
形式的 `lesson` / `steps` / `board` 数据（schema 见 `references/problem-schema.md`）。
不要把参考 HTML 写入或读取出沙箱；模型直接根据返回的数据组装符合 uploader contract
的自包含 HTML，再传给 `generate_game` 的 `html` 字段：

```python
import json
data = {"lesson": {}, "steps": [], "board": {}}
print(json.dumps(data, ensure_ascii=False))
```

不要调用 `python3`、shell 管道或依赖沙箱 cwd 中的 `lib/` 目录；只打印 JSON
数据，不要把临时文件作为交付物。

- `steps[*].content` 里的数值**直接引用 kernel 结果**（用 `K.tex(...)` 输出 LaTeX），模型只负责
  组织讲解文字（按目标语言）。
- `board` 用 kernel 给的曲线 `board` dict、精确点坐标、`param`、`derived` 构造序列、`readouts`、
  `rangeBar`（范围题）/ `constant`（定值题）/ `answerBand`（**形状参数题**，如离心率范围）。
- **形状参数题（滑块=离心率 e 等）**：自然动态量是曲线本身的形状而非动直线/动点时，让滑块=该参数，
  把曲线 `a/b/c`、焦点、动点坐标写成 `@param` 的**表达式字符串**（引擎每帧重绘曲线/焦点/渐近线），
  配 `status` 读数显示不等式状态、`answerBand` 在参数轴高亮答案区间。见 conventions「形状参数题」。
- **可直接照抄的范本**：`scripts/generate.py` 里 6 个 `build_*` 覆盖各类交互范式：
  `ellipse_dot_range`（范围条）、`ellipse_chord_range`、`ellipse_area_max`、
  `ellipse_slopeprod_const`（定值·中心对称）、`parabola_dot_const`（定值·抛物线）、
  `hyperbola_ecc_range`（**形状参数**：滑块=e，曲线随之重绘 + `status` + `answerBand`）。

`scripts/generate.py` 是参考实现；在 agent 中不要把它当作 shell 命令执行。
如需复用其中的构造逻辑，把所需的 Python 函数复制为 `execute_code` 源码，
导入 `analytic_kernel` 和 `conics`，并只打印结构化数据。

## Pitfalls

- Do not use a host interpreter or install packages during a request; the isolated runtime is immutable and network-disabled.
- Do not upload the reference template, Python files, screenshots, or temporary sandbox paths as the final artifact.

## Verification
- kernel 答案 == 答案卡 `lesson.answer` == 末步骤展示值 == **JS 标准位/扫段重算值**，四者一致
  （`build_*` 内已加 `assert`）。
- `rangeBar` 端点来自 kernel 的 `range_over_m`；`constant` 值来自 kernel 的定值。
- Use `execute_code` for deterministic kernel assertions and structural checks on the generated HTML:
  confirm the data island parses, all required board/readout elements are present, and no external
  runtime dependency or temporary file is required.
- Submit the self-contained HTML with `generate_game` and verify the returned artifact result. Do not
  create a local browser server or persist screenshots in the sandbox.

### Delivery
成品通过 `generate_game` 上传并把返回的媒体结果交付给用户。交付前确认没有遗留沙箱临时文件，
也不要把沙箱中的临时文件路径告诉用户。

## 扩展
- **加题型**：在 `analytic_kernel.py` 加目标量函数（写成 m 的表达式）+ 复用 `range_over_m` /
  `is_constant_in_m`；在 `generate.py` 加一个 `build_*`，选定交互范式（范围条 / 定值 / 定点 /
  轨迹 trace / 形状参数 answerBand）。见 `references/conventions.md` 配方表。
- **加曲线**：`conics.py` 已有椭圆/双曲线/抛物线/圆；前端 `board.html` 引擎已支持四类渲染、
  渐近线、准线方向。新曲线在两处各加一份即可。
- **加交互构造**：`board.html` 的 `buildScene` switch 是构造库（`line_through_angle`、
  `intersect_line_conic`、`point_on_conic`、`point_reflect`、`tangent_at`、`foot_perp`…），
  按需扩充并在 schema 文档登记。

## 目录
- `templates/board.html` — 数据驱动模板（通用 2D 渲染器 + 参数引擎 + 数据岛 `__LESSON_DATA__`）
- `lib/conics.py` — 圆锥曲线 sympy 定义库（特殊点 / LaTeX / board dict）
- `lib/analytic_kernel.py` — sympy 精确求解核心（联立·韦达·范围·定值）
- `scripts/generate.py` — 注入模板 + 5 个 build_* 范本 + 批量/单题出题
- `references/problem-schema.md` — 数据格式（board 引擎 schema）
- `references/conventions.md` — 标准式、解法配方表、韦达/换元套路、端点开闭、自检
