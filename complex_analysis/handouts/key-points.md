# 复变 handouts 重要知识点

阅读范围以 `complex_analysis/handouts` 中的复分析主讲义最新版为准：

- `CA-chap1-2026-04-27.pdf`
- `CA-chap2-2026-05-31.pdf`
- `CA-chap3-2026-04-30.pdf`
- `CA-chap4-2026-06-13.pdf`

未纳入主线整理：`bookNumPDEsHandout-*`、`CA-appendices-*`、`appendices.pdf` 主要是数值分析/PDE 或数学附录；`syllabus-complexAnal.pdf` 主要是课程安排。

## 1. 复数、级数与基本复函数

### 1.1 复数域与复数运算

- 复数域 `C` 可看作 `R x R`，乘法为
  $$
  (x+iy)(u+iv)=(xu-yv)+i(xv+yu).
  $$
- `C` 不是有序域：任意有序域中平方非负，但 `i^2=-1`。
- 共轭与模：
  $$
  \bar z=x-iy,\qquad |z|=\sqrt{z\bar z}=\sqrt{x^2+y^2}.
  $$
- 常用恒等式：
  $$
  \overline{z\pm w}=\bar z\pm \bar w,\quad \overline{zw}=\bar z\bar w,\quad
  z^{-1}=\frac{\bar z}{|z|^2}\;(z\ne 0),
  $$
  $$
  |zw|=|z||w|,\quad |z+w|\le |z|+|w|.
  $$

### 1.2 复数列与幂级数

- 复数列收敛等价于实部、虚部分别收敛：
  $$
  z_n\to z \iff \Re z_n\to \Re z,\quad \Im z_n\to \Im z.
  $$
- 正常收敛（normal convergence）推出绝对收敛和局部一致收敛；这是之后逐项积分、逐项求导的基础。
- 幂级数
  $$
  \sum_{n=0}^{\infty}a_nz^n
  $$
  有唯一收敛半径。半径内正常收敛，半径外发散。
- Cauchy-Hadamard 公式：
  $$
  R=\frac{1}{\limsup_{n\to\infty}|a_n|^{1/n}}.
  $$

### 1.3 指数、三角、对数与幂函数

- 指数函数由幂级数定义：
  $$
  e^z=\sum_{n=0}^{\infty}\frac{z^n}{n!}.
  $$
- 核心性质：
  $$
  e^{z+w}=e^ze^w,\qquad e^z\ne 0,\qquad e^{z+2\pi i}=e^z.
  $$
- Euler 公式：
  $$
  e^{iy}=\cos y+i\sin y,\qquad e^{x+iy}=e^x(\cos y+i\sin y).
  $$
- 复三角函数：
  $$
  \cos z=\frac{e^{iz}+e^{-iz}}{2},\qquad
  \sin z=\frac{e^{iz}-e^{-iz}}{2i}.
  $$
- 复对数本质上是多值函数：
  $$
  \log z=\log|z|+i(\arg z+2k\pi),\quad k\in Z.
  $$
- 主值辐角 `Arg z` 在割平面
  $$
  C^- = C\setminus (-\infty,0]
  $$
  上连续；主值对数为
  $$
  \Log z=\log|z|+i\Arg z.
  $$
  它沿负实轴不连续。
- 幂函数通常通过对数分支定义：
  $$
  z^\alpha=\exp(\alpha\Log z)
  $$
  这是主值分支，不是全局单值函数。
- 复数乘法的几何意义：乘以非零复数 `a` 等于旋转 `arg a` 加缩放 `|a|`。因此 `C^*` 与平面上保定向相似变换有自然联系。

## 2. 复微分、Cauchy-Riemann 方程与调和函数

### 2.1 复可微

- 复导数定义：
  $$
  f'(a)=\lim_{z\to a}\frac{f(z)-f(a)}{z-a}.
  $$
- 复可微比实可微强得多：差商极限必须与趋近方向无关。
- 若在点处复可微，则在该点连续。
- 四则运算、链式法则成立。典型导数：
  $$
  (e^z)'=e^z,\quad (\Log z)'=\frac1z,\quad
  (\sin z)'=\cos z,\quad (\cos z)'=-\sin z.
  $$

### 2.2 Cauchy-Riemann 方程

设
$$
f(z)=u(x,y)+iv(x,y),\qquad z=x+iy.
$$
若 `u,v` 实可微，则 `f` 复可微等价于
$$
u_x=v_y,\qquad u_y=-v_x.
$$
此时
$$
f'(z)=u_x+iv_x=v_y-iu_y.
$$

等价地，Jacobian 必须是一个复线性映射：

$$
J_f=
\begin{pmatrix}
u_x&u_y\\
v_x&v_y
\end{pmatrix}
=
\begin{pmatrix}
a&-b\\
b&a
\end{pmatrix}.
$$

### 2.3 解析与共形

- 在开集上处处复可微称为解析/全纯。
- 若 `f` 解析且 `f'(z_0)\ne 0`，则 `f` 在 `z_0` 处共形：局部表现为
  $$
  f(z)\approx f(z_0)+f'(z_0)(z-z_0),
  $$
  即旋转 `arg f'(z_0)` 加缩放 `|f'(z_0)|`。
- 复共轭 `z\mapsto \bar z` 保角但反定向，因此不是全纯函数。
- 解析函数的临界点与局部映射行为在第 4 章进一步由零点重数刻画。

### 2.4 连通性与常值结论

在区域（非空连通开集）上，若解析函数满足以下任一条件，则为常数：

- `f'=0`；
- `Re f` 为常数；
- `Im f` 为常数；
- `|f|` 为常数。

连通性是关键；若定义域不连通，局部常数不必是全局常数。

### 2.5 调和函数

- 若 `f=u+iv` 解析，则 `u,v` 都调和：
  $$
  \Delta u=0,\qquad \Delta v=0.
  $$
- 若 `u` 是调和函数，在适当区域上可寻找调和共轭 `v`，使 `u+iv` 解析。
- 在单连通区域上，调和函数存在全局调和共轭；共轭函数差一个常数。

## 3. 围道积分与 Cauchy 理论

### 3.1 围道积分

对分段光滑曲线 `gamma:[a,b]->C`，
$$
\int_\gamma f(z)\,dz=\int_a^b f(\gamma(t))\gamma'(t)\,dt.
$$

基本性质：

- 对曲线拼接可加；
- 改变参数不改变积分值；
- 反向曲线积分变号；
- ML 估计：
  $$
  \left|\int_\gamma f(z)\,dz\right|\le M\,L(\gamma),
  $$
  其中 `|f|\le M`，`L(gamma)` 是曲线长度。

### 3.2 原函数、闭路积分与路径无关

对连续函数 `f:D->C`，以下三件事密切相关：

- `f` 有原函数 `F'=f`；
- 任意闭曲线 `gamma` 上
  $$
  \oint_\gamma f(z)\,dz=0;
  $$
- 积分只依赖端点，与路径无关。

典型反例：
$$
\oint_{|z|=1}\frac{dz}{z}=2\pi i\ne 0.
$$
所以 `1/z` 在 `C^*` 上没有全局原函数。

### 3.3 Cauchy 积分定理

核心版本：

- 若 `D` 单连通，`f` 在 `D` 上解析，则任意可缩闭曲线 `gamma` 满足
  $$
  \oint_\gamma f(z)\,dz=0.
  $$
- Goursat 定理说明：只假设复可微即可推出三角形边界积分为零，不需要假设 `f'` 连续。
- 在星形域、单连通域、多连通域中，Cauchy 定理有不同形式；多连通情形要考虑外边界和洞的边界贡献。

### 3.4 Cauchy 积分公式

若 `f` 在圆盘闭包附近解析，`|z-a|<r`，则
$$
f(z)=\frac{1}{2\pi i}\oint_{|\zeta-a|=r}
\frac{f(\zeta)}{\zeta-z}\,d\zeta.
$$

意义：解析函数内部的值由边界值完全决定。

推广到导数：
$$
f^{(n)}(z)=\frac{n!}{2\pi i}\oint_{|\zeta-a|=r}
\frac{f(\zeta)}{(\zeta-z)^{n+1}}\,d\zeta.
$$

Cauchy 不等式：
$$
|f^{(n)}(a)|\le \frac{n!M}{r^n},
$$
其中 `M=max_{|\zeta-a|=r}|f(\zeta)|`。

### 3.5 重要推论

- 解析函数自动无穷次复可微。
- Morera 定理：连续函数若闭路积分恒为 0，则解析。
- Liouville 定理：有界整函数必为常数。
- 代数基本定理：非常数复系数多项式在 `C` 中有根。
- Weierstrass 定理：解析函数列若局部一致收敛，则极限解析，且导数也局部一致收敛。
- 幂级数在收敛圆内可逐项求导、逐项积分。

### 3.6 全纯性的等价刻画

对开集上的函数，以下性质在适当连续性/局部条件下等价：

- 复可微，即解析；
- 实可微并满足 Cauchy-Riemann 方程；
- 三角形边界积分为零；
- 局部存在原函数；
- 满足 Cauchy 积分公式；
- 局部可表示为收敛幂级数；
- 在每个完全包含于定义域的圆盘中有 Taylor 展开。

这是复分析的核心刚性：复可微、积分为零、幂级数展开、无穷次可微并不是彼此独立的性质，而是同一个现象的不同表现。

## 4. 幂级数、零点、最大模与 Schwarz 引理

### 4.1 Taylor 展开

若 `f` 在开集 `D` 上解析，则对任意 `c in D` 和 `U_R(c) subset D`，
$$
f(z)=\sum_{n=0}^{\infty}a_n(z-c)^n,\qquad
a_n=\frac{f^{(n)}(c)}{n!}
=\frac{1}{2\pi i}\oint_{|\zeta-c|=\rho}
\frac{f(\zeta)}{(\zeta-c)^{n+1}}\,d\zeta.
$$

注意：

- 收敛半径由复平面中的奇点控制，不只由实轴行为控制。
- 例如 `1/(1+z^2)` 在实轴上无奇点，但 Taylor 半径受 `z=±i` 限制。
- 边界上的收敛行为不能只由半径内结论判断。

### 4.2 初等域与解析对数

- 初等域：定义在其上的任意解析函数都有全局原函数。
- 若 `f` 在初等域上解析且无零点，则存在解析函数 `h` 使
  $$
  f=e^h.
  $$
  这就是解析对数分支。
- 因而也可定义解析的 `n` 次根：
  $$
  H^n=f.
  $$

### 4.3 零点与恒等定理

若 `f` 解析且 `a` 是 `m` 重零点，则局部有分解
$$
f(z)=(z-a)^m\varphi(z),\qquad \varphi(a)\ne 0.
$$

重要结论：

- 非零解析函数的零点在定义域内离散；
- 若两个解析函数在有内聚点的集合上相等，则它们在整个连通区域上相等；
- 若解析函数在某点所有阶导数都相等，则在连通区域上相等；
- 若两个解析函数乘积恒为 0，则其中一个恒为 0。

### 4.4 局部映射行为

若 `f(0)=0` 且 0 是 `m` 重零点，则局部存在共形映射 `g`，使
$$
f(z)=g(z)^m.
$$

直观意义：

- 简单零点附近局部像一个共形映射；
- `m` 重零点附近角度被乘以 `m`，局部呈 `m` 重覆盖。

### 4.5 开映射、最大模与最小模

- 开映射定理：非常数解析函数把开集映成开集。
- 最大模原理：非常数解析函数的模不能在区域内部取到最大值。
- 紧集版本：若 `K subset D` 紧，则最大模出现在边界。
- 最小模原理：非常数解析函数若在内部取得局部最小模，则该点必须是零点。

这些结论是很多估计和唯一性论证的核心。

### 4.6 Schwarz 引理与圆盘自同构

Schwarz 引理：若 `f:E->E` 解析且 `f(0)=0`，则
$$
|f'(0)|\le 1,\qquad |f(z)|\le |z|.
$$
若等号在非零点成立，或 `|f'(0)|=1`，则
$$
f(z)=\zeta z,\qquad |\zeta|=1.
$$

单位圆盘自同构：
$$
\operatorname{Aut}(E)=
\left\{
z\mapsto e^{i\theta}\frac{z-a}{\bar a z-1}:\theta\in[0,2\pi),\ a\in E
\right\}.
$$

Schwarz-Pick 不等式：
$$
\frac{|f'(z)|}{1-|f(z)|^2}\le \frac{1}{1-|z|^2}.
$$
等号在一点成立当且仅当 `f` 是圆盘自同构。

## 5. 解析延拓与 Schwarz 反射

### 5.1 解析延拓

若 `F:G->C` 解析，`G superset D`，且 `F|_D=f`，则 `F` 是 `f` 的解析延拓。

唯一性：

- 若延拓存在，且原函数定义在有内聚点的集合上，则解析延拓唯一。
- 这是恒等定理的直接应用。

### 5.2 Schwarz 反射原理

若区域关于实轴对称，`f` 在上半部分解析、在边界实轴上取实值并连续到边界，则可通过
$$
F(z)=\overline{f(\bar z)}
$$
延拓到下半部分。

圆周版本也类似：若边界圆上 `|f|` 为常数，可通过反演和共轭给出解析延拓。

## 6. 孤立奇点、Laurent 级数与亚纯函数

### 6.1 孤立奇点

若 `c` 不在定义域中，但某个穿孔圆盘
$$
\dot U_r(c)=\{z:0<|z-c|<r\}
$$
包含在定义域内，则 `c` 是孤立奇点。

三类孤立奇点：

- 可去奇点；
- 极点；
- 本性奇点。

### 6.2 可去奇点

Riemann 可去奇点定理：

`c` 是可去奇点，当且仅当 `f` 在某个穿孔邻域内有界。

常用等价条件：

- `f` 在 `c` 附近有界；
- `lim_{z->c} f(z)` 存在；
- `lim_{z->c}(z-c)f(z)=0`。

典型例子：
$$
\frac{\sin z}{z}
$$
在 `0` 处是可去奇点。

### 6.3 极点与阶数

`c` 是 `k` 阶极点，当且仅当局部可写为
$$
f(z)=\frac{h(z)}{(z-c)^k},\qquad h(c)\ne 0.
$$

等价地：

- `|f(z)| -> infinity` 当 `z->c`；
- `1/f` 在 `c` 处有 `k` 重零点；
- 存在常数 `M1,M2>0`，使
  $$
  M_1|z-c|^{-k}\le |f(z)|\le M_2|z-c|^{-k}.
  $$

阶数运算：
$$
\operatorname{ord}(fg;c)=\operatorname{ord}(f;c)+\operatorname{ord}(g;c),
$$
$$
\operatorname{ord}(f/g;c)=\operatorname{ord}(f;c)-\operatorname{ord}(g;c).
$$

### 6.4 本性奇点

Casorati-Weierstrass 定理：

若 `c` 是本性奇点，则任意穿孔邻域的像在 `C` 中稠密。

分类的映射行为：

- 可去奇点：某穿孔邻域内有界；
- 极点：趋于无穷；
- 本性奇点：任意穿孔邻域的像在 `C` 中稠密。

### 6.5 Laurent 分解与 Laurent 级数

环域
$$
A_c(r,R)=\{z:r<|z-c|<R\}.
$$

环域上的 Cauchy 公式：
$$
f(z)=\frac{1}{2\pi i}\oint_{|\zeta-c|=R}
\frac{f(\zeta)}{\zeta-z}\,d\zeta
-

\frac{1}{2\pi i}\oint_{|\zeta-c|=r}
\frac{f(\zeta)}{\zeta-z}\,d\zeta.
$$

Laurent 分解：
$$
f(z)=g(z)+h(z),
$$
其中 `g` 是正则部分，`h` 是主部。

Laurent 级数：
$$
f(z)=\sum_{j=-\infty}^{+\infty}a_j(z-c)^j,
$$
系数唯一：
$$
a_j=\frac{1}{2\pi i}\oint_{|\zeta-c|=\rho}
\frac{f(\zeta)}{(\zeta-c)^{j+1}}\,d\zeta.
$$

### 6.6 用 Laurent 级数分类奇点

设
$$
f(z)=\sum_{j=-\infty}^{+\infty}a_j(z-c)^j
$$
是 `c` 附近穿孔圆盘中的 Laurent 展开。

- 可去奇点：
  $$
  a_j=0\quad (j<0).
  $$
- `k` 阶极点：
  $$
  a_{-k}\ne 0,\qquad a_j=0\quad (j<-k).
  $$
- 本性奇点：负幂项无限多个非零。

### 6.7 周期解析函数与 Fourier 级数

若 `f` 是水平条带上的 1-周期解析函数，则通过
$$
q(z)=e^{2\pi iz}
$$
可把条带映到环域，从 Laurent 展开得到 Fourier 展开：
$$
f(z)=\sum_{n=-\infty}^{+\infty}a_ne^{2\pi inz},
$$
其中
$$
a_n=\int_0^1 f(z)e^{-2\pi inz}\,dx.
$$

## 7. 扩充复平面、无穷远点与亚纯函数

### 7.1 扩充复平面

$$
\widehat C=C\cup\{\infty\}.
$$

研究 `infinity` 处的行为，转化为研究
$$
\hat f(z)=f(1/z)
$$
在 `0` 处的奇点。

### 7.2 整函数在无穷远点处的分类

- 整函数在 `infinity` 处是可去奇点，当且仅当它是常数。
- 整函数在 `infinity` 处是极点，当且仅当它是非常数多项式。
- `e^z` 在 `infinity` 处是本性奇点。

### 7.3 亚纯函数

亚纯函数：除离散极点集外解析，并且所有奇点都是极点。

重要结论：

- 有理函数在 `C` 与 `\widehat C` 上都是亚纯函数。
- `cot(pi z)` 在 `C` 上亚纯，但在 `\widehat C` 上不是亚纯，因为极点在无穷远处有聚点。
- `\widehat C` 上的亚纯函数当且仅当有理函数：
  $$
  f(z)=\frac{P(z)}{Q(z)}.
  $$

### 7.4 自同构与函数代数结构

- 整个复平面的全局共形自同构只有仿射映射：
  $$
  \operatorname{Aut}(C)=\{z\mapsto az+b:a\in C^*,\ b\in C\}.
  $$
- 开集上的解析函数 `H(Omega)` 构成含幺交换环。
- 区域上的解析函数环 `O(D)` 是整环。
- 区域上的亚纯函数 `M(D)` 是域。

## 8. 复习时最应抓住的主线

### 8.1 一条核心等价链

复可微
$$
\Longleftrightarrow
$$
Cauchy-Riemann
$$
\Longrightarrow
$$
Cauchy 定理
$$
\Longrightarrow
$$
Cauchy 积分公式
$$
\Longrightarrow
$$
无穷可微和 Taylor 展开
$$
\Longleftrightarrow
$$
解析函数的刚性。

Morera 定理补上了积分条件到解析性的反向推理。

### 8.2 最常用公式

$$
f'(a)=\lim_{z\to a}\frac{f(z)-f(a)}{z-a}
$$

$$
u_x=v_y,\qquad u_y=-v_x
$$

$$
\int_\gamma f(z)\,dz=\int_a^b f(\gamma(t))\gamma'(t)\,dt
$$

$$
\oint_{|z-a|=r}\frac{dz}{z-a}=2\pi i
$$

$$
f(z)=\frac{1}{2\pi i}\oint\frac{f(\zeta)}{\zeta-z}\,d\zeta
$$

$$
f^{(n)}(z)=\frac{n!}{2\pi i}\oint
\frac{f(\zeta)}{(\zeta-z)^{n+1}}\,d\zeta
$$

$$
f(z)=\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(z-a)^n
$$

$$
f(z)=\sum_{j=-\infty}^{+\infty}a_j(z-c)^j,\qquad
a_j=\frac{1}{2\pi i}\oint
\frac{f(\zeta)}{(\zeta-c)^{j+1}}\,d\zeta
$$

### 8.3 常见易错点

- `Arg` 和 `Log` 需要选分支；不能在 `C^*` 上定义全局连续的单值对数。
- `1/z` 在 `C^*` 上解析但没有全局原函数。
- Cauchy 定理需要拓扑条件；洞会贡献非零积分。
- 幂级数的收敛半径由复平面奇点控制，不只看实轴。
- 非零解析函数的零点离散；若零点有内聚点，则函数恒为零。
- 最大模原理只对非常数解析函数给出内部不能取最大值；紧集最大值出现在边界。
- 可去奇点、极点、本性奇点最稳的判别方法是看 Laurent 负幂部分。
- 无穷远点的奇点通过 `f(1/z)` 在 0 处的奇点判断。
