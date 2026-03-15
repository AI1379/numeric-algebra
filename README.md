# Homeworks

各种课程的代码仓库合集。注意因为依赖了 Lua 所以只能用 LuaLaTeX 编译。

## Common

常见 LaTeX 模板。包含 [common/macros.tex](common/macros.tex)，定义了常用的数学算子、矩阵缩写和作业环境。

在作业中可以通过 `\input{../../common/macros.tex}` 引用。

## Tools

- `flatten_tex.py`: 用于将 .tex 文件中的 `\input` 特性展开，生成独立的 .tex 文件（便于投稿或分享）。

    ```bash
    python flatten_tex.py <input.tex> [output.tex]
    ```

## Numeric Algebra

2026 春夏数值代数

## Complex Analysis

2026 春夏复变函数
