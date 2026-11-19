![](https://files.mdnice.com/user/238337/36b2460a-019e-42a2-a086-bdcb92fb8195.jpg)

---

最近在摸索我校的机器学习课程 CSC311，顺着 KNN 看到了一些高维几何里的现象。KNN 的思路很直接：计算测试样本与训练样本之间的 Euclidean distance，找出最近的几个邻居，再利用它们的标签进行预测。但在一些常见的高维分布下，点与点之间的距离会趋于集中，相对差异越来越小。维度增加以后，我们在低维中习惯的“远近”，未必还那么好用。

这与 Curse of Dimensionality 有关。这里先看一个相关的覆盖问题：给定 $\varepsilon>0$，在单位超立方体 $[0,1]^n$ 中放置训练点，要求任意一点到最近训练点的距离都不超过 $\varepsilon$。这相当于以训练点为球心，用半径为 $\varepsilon$ 的球覆盖整个立方体。

记 $n$ 维单位球的体积为 $v_n$，则半径为 $\varepsilon$ 的球体积为 $v_n\varepsilon^n$。如果 $N$ 个球能够完成覆盖，由体积的次可加性，立刻得到
$$
1\leq Nv_n\varepsilon^n
\iff
N\geq\frac{1}{v_n\varepsilon^n}
$$
注意到，这只是必要条件。我们把每个球的体积都算进去了，其中互相重叠、超出立方体的部分并不能提供额外的覆盖。即便如此，固定 $n$ 后，这个下界也已经按 $\varepsilon^{-n}$ 增长：覆盖半径缩小一半，下界就变成原来的 $2^n$ 倍。二十维时是一百多万倍，多少有点不讲道理（

接下来计算 $v_n$。先把记号交代清楚：在 $\mathbb R^n$ 中，令
$$
\|x\|_2=\left(\sum_{i=1}^n x_i^2\right)^{1/2}
\qquad
B_n(r)=\{x\in\mathbb R^n:\|x\|_2\leq r\}
$$
这里的 $B_n(r)$ 就是以原点为球心、半径为 $r>0$ 的闭球。一维时是区间，二维时是圆盘，三维时是通常意义下的球。更高维照着同一个定义往下写就行，暂时不必为难自己的空间想象力 (x

用 $\lambda^n$ 表示 $n$ 维 Lebesgue measure，定义
$$
V_n(r)=\lambda^n(B_n(r))=\int_{B_n(r)}1\,dx
\qquad
v_n=V_n(1)
$$
这样，长度、面积和体积就有了统一的表述。又因为 $B_n(r)=rB_n(1)$，由 Lebesgue measure 的缩放性质 $\lambda^n(rA)=r^n\lambda^n(A)$，有
$$
V_n(r)=v_nr^n.
$$
半径的依赖已经确定，剩下的计算都在常数 $v_n$ 上。这里用一个很经典的办法：找一个既能按坐标分解、又能按半径分层的积分。Gaussian integral 恰好同时具有这两个结构。令
$$
I_n=\int_{\mathbb R^n}e^{-\|x\|_2^2}\,dx.
$$
注意到 $\|x\|_2^2=x_1^2+\cdots+x_n^2$，所以指数里的和可以拆成乘积。被积函数非负，由 Tonelli 定理，
$$
I_n
=\prod_{i=1}^n\int_{-\infty}^{\infty}e^{-x_i^2}\,dx_i
=I_1^n.
$$
先把一维结果补上。$e^{-x^2}$ 没有初等原函数，但把积分平方后，就可以在 $\mathbb R^2$ 中用极坐标计算。令 $x=r\cos\theta,\ y=r\sin\theta$，Jacobian 的绝对值为 $r$，于是
$$
\begin{aligned}
I_1^2
&=\iint_{\mathbb R^2}e^{-(x^2+y^2)}\,dx\,dy\\
&=\int_0^{2\pi}\int_0^\infty e^{-r^2}r\,dr\,d\theta\\
&=2\pi\int_0^\infty re^{-r^2}\,dr
=\pi.
\end{aligned}
$$
由于 $I_1>0$，得到 $I_1=\sqrt\pi$，从而
$$
\boxed{I_n=\pi^{n/2}}.
$$
另一方面，$e^{-\|x\|_2^2}$ 在同一个球面上取值相同，因此也可以把空间按半径分层。**这里需要带上每一层的体积权重。** $dr$ 只记录厚度，半径不同的球壳，即使厚度相同，体积也不同。

具体地，对 $0\leq a<b$，球壳的体积为

$$
\begin{aligned}
\lambda^n\{x:a<\|x\|_2\leq b\}
&=V_n(b)-V_n(a)\\
&=v_n(b^n-a^n)
=\int_a^b nv_nr^{n-1}\,dr.
\end{aligned}
$$

这就给出了按半径积分时的密度 $nv_nr^{n-1}$，也就是常写的 $dV_n(r)=nv_nr^{n-1}\,dr$。在三维里，它正好是 $4\pi r^2\,dr$；熟悉的“球面面积乘以厚度”，背后就是这个关系。因此，
$$
I_n
=\int_0^\infty e^{-r^2}\,dV_n(r)
=nv_n\int_0^\infty r^{n-1}e^{-r^2}\,dr.
$$
前一个积分可以严格理解为关于 $V_n$ 的 Lebesgue–Stieltjes 积分。作代换 $t=r^2$，利用 $r^{n-1}\,dr=\tfrac12t^{n/2-1}\,dt$，得到
$$
I_n=\frac{nv_n}{2}\int_0^\infty t^{n/2-1}e^{-t}\,dt.
$$
现在出现的就是 Gamma function。回忆它的定义及分部积分给出的递推关系：
$$
\Gamma(s)=\int_0^\infty t^{s-1}e^{-t}\,dt,
\qquad
\Gamma(s+1)=s\Gamma(s),
\qquad s>0.
$$
由 $\Gamma(1)=1$ 可得，对非负整数 $m$，有 $\Gamma(m+1)=m!$，所以它可以看作阶乘向非整数情形的推广。这里取 $s=n/2$，恰好有
$$
I_n
=\frac{nv_n}{2}\Gamma\left(\frac n2\right)
=v_n\Gamma\left(\frac n2+1\right).
$$
与按坐标计算得到的 $I_n=\pi^{n/2}$ 比较，就解出了单位球的体积：
$$
\boxed{v_n=\frac{\pi^{n/2}}{\Gamma\left(\frac n2+1\right)}}.
$$
因此，半径为 $r$ 的 $n$ 维 Euclidean 球的体积为
$$
\boxed{V_n(r)=\frac{\pi^{n/2}}{\Gamma\left(\frac n2+1\right)}r^n}.
$$
整段计算的关键，是同一个 Gaussian 同时具有乘积结构与径向对称性：前者让我们把积分算出来，后者把未知的 $v_n$ 带进来。两边一对，几何常数就确定了。

顺带看一下维度的影响。由 Gamma function 的递推关系，
$$
\frac{v_{n+2}}{v_n}=\frac{2\pi}{n+2}.
$$
因此维度足够大后，每增加两维，单位球的体积就会缩小；分别考察奇数维和偶数维，还能直接得到 $v_n\to0$。这里固定的是半径，维度变多并不意味着单位球的体积会一直变大。

回到最初的覆盖问题，现在有了明确的下界：
$$
\boxed{N\geq\frac{\Gamma\left(\frac n2+1\right)}{\pi^{n/2}\varepsilon^n}}.
$$
这是覆盖整个立方体的必要条件，描述的是高维空间中维持局部样本密度的代价。开头提到的距离集中，则需要另行指定数据分布来分析。两者都与高维几何有关，但这段体积计算直接回答的是前一个问题。


