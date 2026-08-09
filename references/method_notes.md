# 方法说明与方法学注记

本 skill 分析方法的物理约定、单位链与使用边界。所有结论行(`CONCLUSION:`)的数值含义以此为准。

## 1. D 单位换算链(MSD)

```
MSD(t) = <|r(t) - r(0)|^2>         [A^2], 第一帧为参考
D = slope / dims                   [A^2/ps]
D [cm^2/s] = slope × 1e-4 / dims   dims = 6 (3D) / 4 (xy 2D) / 2 (z 1D)
```

- 时间轴:`t [ps] = 帧号 × dt_fs [fs] / 1000`;dt_fs 优先级:l1 `meta.json` > dump TIMESTEP 差推算 > `--timestep-fs`
- 拟合窗口默认轨迹 20%–90%(跳过弹道区与末端统计噪声);`--diffusive-check` 输出 log-log 斜率,≈1 才是爱因斯坦区,偏离 >0.2 结果不可信
- **必须先 unwrap**:对 PBC 包裹坐标直接算 MSD 会得到虚假平台;l1 产物 `*_uw.dump` 已 unwrap
- **COM 漂移**:长轨迹/慢扩散参考系下,MSD 会因整体漂移虚高;`l1_traj_prep --unwrap com --com-ref "1 2"` 或 `l3_msd --com-subtract`(慢扩散层作参考更稳)
- unwrap 算法(frac diff-round-cumsum):对已 unwrap 输入天然幂等,支持逐帧盒子

## 2. RDF 与配位数

- `InterRDF` 归一化基于全盒体积与逐帧密度(NPT 变体积安全)
- `CN(r) = 4πρ ∫ g(r') r'² dr'`,ρ = 目标原子数 / 平均体积
- 峰检测:第一峰搜索窗 [1.5, 5.0] Å,第一谷 [r_pk+0.3, r_pk+1.5] Å;峰须显著(>2× 窗口均值)才报告
- 区域过滤默认第一帧静态;`--per-frame` 本 MVP 未实现(RDF 的 InterRDF 不支持逐帧换 selection)

## 3. 氢键

- 几何判据(水体系惯例):供体-受体距离 < 3.5 Å,供体-H-受体角 > 150°;参数全暴露(`--distance`/`--angle`)
- `C(t)` = 连续自相关:每 (donor,acceptor) 对把存在帧拆成连续 run,C(t) = Σ_runs max(L−t, 0) / Σ_runs L——即"t=0 存在且连续存活到 t"的概率;**有限轨迹中静态键的 C(t) 也是线性衰减 (L−t)/L,不是 1**
- τ_hb:单指数 log-linear 拟合(掩码 C(t)∈(0.01, 1]);双指数 curve_fit(带 bounds),τ_avg = A 加权
- 氢键事件表的 donor/acceptor 是 **universe 0-based 索引**(不是 LAMMPS id)

## 4. 密度分布与水层

- 1D 剖面:每帧直方图 + PBC wrapping(mod 盒长)+ 逐帧 bin 体积归一(NPT 安全);不用 MDAnalysis `DensityAnalysis`(3D 网格化,对 1D 剖面过重)
- 数密度 [Å⁻³];质量密度 [g/cm³] = u/Å³ × 1.66054(需元素映射)
- 水层识别:平滑 ρ(z) 后取**显著**局部极小(深度 > 30% 相邻较低峰),过滤噪声纹波;层数 = 极小点个数 + 1

## 5. 序参量(l2_order)

- q_l = √(4π/(2l+1) · Σ_m |⟨Y_lm(r̂)⟩|²),球谐用 `scipy.special.sph_harm_y(n, m, theta, phi)`(scipy ≥1.15;旧 `sph_harm` 在 1.17 移除)
- **邻居必须是同元素**(如 O-O),不是全原子——否则水分子的 H 会被当成 O 的第一壳层
- 最小镜像 PBC(正交盒);第一壳层截止默认 3.5 Å
- 冰/液水判据参考:fcc q6 = 0.5745(Steinhardt 1983)、理想四面体 q_tet = 1(平面正方形 0.5)、液水 q6 ≈ 0.2-0.4
- LSI = 最近 4 邻居距离的 rms 偏差;液水 ~0.4-0.7 Å²,冰更窄
- **帧间行序不稳定**:未 `sort id` 的 dump 帧间原子行序可变,中心原子一律按 LAMMPS id 定位(实测真实轨迹触发过)

## 5b. VACF/VDOS(l3_vacf_vdos)

- **必须 dump 带速度列**(`vx vy vz`),解析层自动读取;缺失时报错并给 LAMMPS dump 命令示例
- VACF(τ) = ⟨v(t0+τ)·v(t0)⟩/⟨v(0)²⟩,全原子 × 全时间原点平均,归一化 VACF(0)=1
- VDOS = Re FFT(VACF × Hanning);频率轴 `rfftfreq(n, d_ps)` 单位已是 THz(1/ps = 1 THz);cm⁻¹ = THz × 33.356
- 参考峰:O-H 伸缩 ~108 THz(3600 cm⁻¹)、平动/摇摆 10-25 THz、氢键带 5-7 THz(依赖势函数,MACE/AIMD 值有偏移)

## 5c. TCF(l3_tcf)

- **驻留时间**:连续 residence TCF(Impey 1983 系综平均)C(t) = ⟨h(t0)h(t0+1)…h(t0+t)⟩/⟨h(t0)⟩;永不离开的分子 C(t) ≡ 1(不是 l2_hbonds 的键寿命衰减语义,两者勿混)
- **偶极弛豫**:C(t) = ⟨μ̂(t0+t)·μ̂(t0)⟩,τ 单指数 log-linear 拟合
- 两者都需要水分子识别(molecules.py:O 与最近 2 个 H 在 1.3 Å 内配对)与 unwrapped 坐标

## 5d. 界面分析(l4_interface)

- 偶极取向:cosθ = μ̂·ẑ,直方图 + 均值;界面水物理:cosθ 偏 ±1 表示偶极指向/背向表面,0 平躺
- 吸附质 2D 密度:x/y PBC wrap 后直方图,按 bin 面积与帧数归一 (Å⁻²)
- 层性质:与 l2_density 相同的显著极小分层,另加**稀疏层过滤**(n_O < 2% 总量的噪声谷丢弃)

## 6. 平衡判断(log.lammps)

- thermo 块合并:多块(列集不同)按列名交集合并,保第一块列序
- 漂移判据:`|斜率 × 窗口宽| < drift_tol × 残差σ`;默认 tol=0.5(纯噪声有限样本的斜率散度 ~ σ√(12/N),tol 须 > 该值才不误判)
- 分段均值:--nblocks 等分,输出块均值 ± σ,首尾块差目检

## 6. 区域选择语义

- 静态(默认):第一帧掩码,原子身份跨帧固定(MSD 必须);`--per-frame` 仅密度支持
- box/sphere/layer 参数见 `region.py --help`;`--layer-auto` 从 marker 原子 z 自动检测双表面(界面 = surf±eps;体相 = 两表面间扣除 eps+margin)
- 密度剖面与 RDF 的区域 g(r) 归一化差异:见 `references/md_analyzer_notes.md` 规避项 #5

## 7. 版本兼容记录(实测)

| 依赖 | 实测版本 | 注意 |
|---|---|---|
| MDAnalysis | 2.10.0 | `HydrogenBondAnalysis` 位于 `MDAnalysis.analysis.hydrogenbonds`(旧 `hbonds` 路径 3.0 移除);selection 必须传字符串 |
| numpy | 2.4.6 | — |
| scipy | 1.17.1 | `sph_harm_y(n, m, theta, phi)`;旧 `sph_harm` 已移除 |
| matplotlib | 3.11.1 | Agg 后端强制 |
| Python | 3.11.15 | uv 托管 |
