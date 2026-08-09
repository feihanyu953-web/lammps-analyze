# MD-Analyzer 移植清单

本文档记录从 [Maygon0303/MD-Analyzer](https://github.com/Maygon0303/MD-Analyzer)(MIT License)移植到本 skill 的实现清单。

- **来源**:`Maygon0303/MD-Analyzer` @ commit `924367eda5df2bf905bcecd4268af42cbd4410a7`
- **许可**:MIT License(来源仓库根 LICENSE),移植代码保留来源署名于本文件与 SKILL.md Acknowledgements
- **日期**:2026-08-09

## 1. 可直接移植(修 bug 后逐字级采用)

| # | 来源函数 | 移植到 | 移植要点 / 修正 |
|---|---|---|---|
| 1 | `core/selector.py::filter_atoms_spatial` | `scripts/region.py` box/sphere 过滤 | 纯 numpy 三轴与运算 / 球壳距离过滤。**必须修原版 bug**:UI 写 `ref_x_min` 前缀键、过滤读 `x_min`,自定义区域静默失效 → 我们统一单一参数 dict |
| 2 | `analyzers/msd/engine.py::_detect_zones` + `_build_auto_mask` | `scripts/region.py` `--region layer auto` | 双表面自动检测:marker 原子 z 中位切分 → 下表面上沿/上表面下沿 → interface=[surf±ε]、bulk=[surf_low+cutoff+margin, surf_up−cutoff−margin];gap<0 或 >100 Å 报非双表面系统 |
| 3 | `analyzers/msd/engine.py::_compute_msd` | `scripts/l3_msd.py` COM 扣除 | 参考帧 + 每帧各减自己 COM(`pos − pos.mean(axis=0)`) |
| 4 | `analyzers/msd/engine.py::_fit_diffusion` | `scripts/l3_msd.py` 拟合段 | polyfit + R²。原版只切拟合起点 → 补 calc_msd 的 20%-90% 双端窗口 |
| 5 | `analyzers/hbond/engine.py::_compute_ct` + `_extract_runs` | `scripts/l2_hbonds.py` C(t) | 逐 (donor,acceptor) 对把连续帧拆 run,C(t)=Σ max(L−t,0)/ΣL(连续自相关,非间歇) |
| 6 | `analyzers/hbond/engine.py::_fit_lifetime` | `scripts/l2_hbonds.py` 寿命拟合 | log-linear 单指数 τ=−1/b(掩码 ct∈(0.01,1.0]);双指数 curve_fit 带 bounds、τ1/τ2 交换、τ_avg=A 加权 |
| 7 | `analyzers/hbond/engine.py::_setup_interface_regions` / `_z_region` / `_classify_events` | `scripts/l2_hbonds.py` 界面事件分类 | 按 donor/acceptor Z 判据分类事件到 interface/bulk/transition |
| 8 | `analyzers/rdf/engine.py` CN 计数 + `_find_peaks` | `scripts/l2_rdf_cn.py` | CN 用 `distance_array` 逐 bin 累计计数 ÷ (n_frames × n_ref);峰检测:第一峰 r∈[1.5,5.0]、第一谷 [r_pk+0.3, r_pk+1.5] |
| 9 | 帧采样 `step = max(1, n_total//n_sample)` | 所有 l2/l3 模块 | `range(0, n_total, step)[:n_sample]` |
| 10 | 结果 dict 约定 + error 字段 + 导出头 | 输出契约 | `"t(ps)  MSD(A^2)  # key"` 等;error 字段贯穿 |
| 11 | `core/density.py` histogram 累加骨架 | `scripts/l2_density.py` | **必须补**:PBC wrapping(mod 盒长)与数密度归一化(原版纯计数、无 PBC、可视化用途) |

## 2. 只借鉴设计

- Analyzer 插件接口(ANALYZER_META + compute/render 四件套)→ 本 skill 的"每分析器一个独立 CLI 子命令 + 统一输入输出契约"
- AppState"core 字段 + analyzer 私有 namespace"职责隔离 → 共享库(scripts/dump_io.py 等)只被分析模块依赖,反向禁止
- 区域三模式(Box/Spherical/Interface)与"先看密度图再选区域"工作流 → SKILL.md 数据准备章节
- `format="LAMMPSDUMP"` 硬编码 → 我们按文件扩展名/格式参数分派

## 3. 明确不要照抄(MD-Analyzer 的 bug/半成品)

1. **区域键名错位 bug**(UI 写 `ref_x_min`,过滤读 `x_min`,which 参数从未使用)→ 自定义区域静默退化全盒。我们统一键名,测试锁定
2. **RDF/hbond 的 interface_mode 空壳**(checkbox 存在但引擎不检测 zone)→ 要么实现要么去掉,不允许空壳
3. **`filter_atoms_by_region` 的 `ag.ids[mask]−1` 零基假设**(ids 不连续时错位)→ 我们存 id 列,不假设连续
4. **`_unwrap_positions` 三重 Python 循环 + 固定 frame-0 盒子 + >L/2 阈值法** → 不采用。用 calc_msd.py 的 frac diff-round-cumsum 向量化实现(逐帧盒子、任意位移、对已 unwrap 输入幂等)。对比:

   | | MD-Analyzer 阈值法 | calc_msd.py frac-round-cumsum |
   |---|---|---|
   | 坐标 | 绝对笛卡尔 | 分数坐标 |
   | 修正量 | Δ>L/2 判定 | round(Δfrac) = 跨像次数 |
   | 实现 | 逐原子×逐帧 Python 循环 | 全向量化 |
   | 盒子 | 固定 frame-0 | 逐帧盒子 |
   | 大位移 | >1.5L 修正错误 | round 任意位移 |
   | 已 unwrap 输入 | 真实位移恰>L/2 误修正 | 天然幂等 |

5. **RDF 区域 g(r) 用全盒体积归一**(区域 RDF 量级近似)→ 我们按区域体积归一或文档标注限制
