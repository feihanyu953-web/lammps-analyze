---
name: lammps-analyze
description: LAMMPS/MACE-MD 轨迹本地后处理(RDF/配位数、密度分布与水层识别、氢键统计与寿命、MSD 与扩散系数、log 平衡检查)。输入 dump.lammpstrj + data.lammps,输出 CSV+PNG+一行结论。需要结构渲染转接 ovito-mcp、发表级曲线转接 origin-mcp。当用户要算 LAMMPS 轨迹的 RDF、扩散系数、密度、氢键、或检查 log 平衡时使用。
---

# lammps-analyze

LAMMPS / MACE-MD 轨迹**本地**后处理 skill。分析全部在本机执行,轨迹通过 cluster-connect 拉取。

```
dump.lammpstrj ─┐
data.lammps  ───┼─→ 【解析层 · 纯 numpy】──→ *_uw.dump ──→ 【分析层 · numpy + MDAnalysis】──→ CSV + PNG + CONCLUSION
log.lammps   ───┘   dump_io: 解析/unwrap/去漂移           l2/l3: RDF/密度/氢键/MSD
```

与 [vaspkit-remote](../vaspkit-remote/SKILL.md) 的定位区别:**vaspkit-remote 是远程执行**,本 skill 是**远程拉取 + 本地分析**。

## 环境引导(首次自动)

首次调用先执行(幂等;已就绪直接打印 ready 退出):

```powershell
python <skill-dir>/scripts/bootstrap_env.py
```

- 在 `<skill-dir>/.venv` 建独立环境(uv + Python 3.11),安装 numpy/matplotlib/MDAnalysis,**不污染其他环境**
- `.venv/.ready` 标记存在即跳过;`--force` 重建;`--skip-install` 只建 venv 排障
- 之后所有脚本用 `<skill-dir>/.venv/Scripts/python.exe`(POSIX 为 `bin/python`)执行,不要用系统 python

## 前置依赖

| 依赖 | 用途 | 必需 |
|---|---|---|
| uv | 环境引导 | 是(缺失时 bootstrap 报错并给安装指引) |
| Python 3.11 | 运行时(uv 自动托管) | 是 |
| cluster-connect | 从集群 pull 轨迹 | 拉轨迹时是,纯本地分析否 |
| ovito-mcp / origin-mcp | 下游转接(渲染/发表级出图) | 否 |

## 数据准备

1. **确认输入文件**:`dump.lammpstrj`(必需);`data.lammps`(推荐,提供类型→元素映射)。dump 要求见 `references/lammps_dump_format.md`:列名驱动、id 可跳号、三斜不支持、`x y z`(需 unwrap)vs `xu yu zu`(已 unwrap)
2. **从集群拉取**(cluster-connect):
   ```powershell
   python <cc-dir>/scripts/cluster.py pull <profile> run2/dump.lammpstrj ./
   python <cc-dir>/scripts/cluster.py pull <profile> run2/data.lammps ./
   ```
3. 缺 data.lammps 时用 `--elements-map "1:O 2:H ..."` 兜底(元素名→type 号需从 LAMMPS input 的 `create_atoms`/`set type` 或 `group` 定义确认)

## 第一步:选择分析路径

| 需求 | 路径 |
|---|---|
| 检查 log 平衡、温度/能量水平 | `l1_log_parse.py --log log.lammps` |
| unwrap / COM 去漂移 / 重采样 / 区域过滤 | `l1_traj_prep.py`(后续 l3 的推荐前置) |
| RDF g(r) + 配位数 | `l2_rdf_cn.py` |
| 密度分布 / 界面水层数 | `l2_density.py --layers` |
| 氢键数目 + 寿命 C(t) | `l2_hbonds.py` |
| 结构序参量(q4/q6/四面体/LSI,冰水区分) | `l2_order.py --elements O` |
| MSD + 扩散系数 | 先 `l1_traj_prep`(unwrap)再 `l3_msd.py` |
| VACF → VDOS 振动谱(需 dump 带 vx vy vz) | `l3_vacf_vdos.py` |
| 界面水驻留时间 / 偶极弛豫 | `l3_tcf.py --mode both --region layer ...` |
| 偶极取向 / 吸附质 2D 分布 / 水层性质 | `l4_interface.py --mode all` |

## 调用模板

```powershell
$py = "<skill-dir>/.venv/Scripts/python.exe"

# 1. log 平衡检查
& $py <skill-dir>/scripts/l1_log_parse.py --log log.lammps --name run2

# 2. 轨迹预处理(unwrap + 重采样;输出 *_uw.dump + *_meta.json)
& $py <skill-dir>/scripts/l1_traj_prep.py --dump dump.lammpstrj --data data.lammps `
    --start 0 --stop -1 --stride 10 --unwrap pbc --name run2

# 3. RDF + CN(O-O 与 O-H 对;配位截止 3.5 A)
& $py <skill-dir>/scripts/l2_rdf_cn.py --dump run2_uw.dump --data data.lammps `
    --pairs "O:O O:H" --cn-cut 3.5

# 4. 密度分布 + 水层识别
& $py <skill-dir>/scripts/l2_density.py --dump run2_uw.dump --data data.lammps `
    --axis z --layers --density-type mass

# 5. 氢键统计 + 寿命(水体系默认判据 3.5 A / 150 deg)
& $py <skill-dir>/scripts/l2_hbonds.py --dump run2_uw.dump --data data.lammps `
    --lifetime-method double

# 6. MSD + 扩散系数(meta.json 自动提供 dt 与元素映射)
& $py <skill-dir>/scripts/l3_msd.py --dump run2_uw.dump --data data.lammps `
    --elements "O K" --dims all --diffusive-check

# 7. 结构序参量(O 原子,q4/q6/四面体/LSI,冰水区分)
& $py <skill-dir>/scripts/l2_order.py --dump run2_uw.dump --data data.lammps `
    --elements O --order all --lsi

# 8. VACF → VDOS(需要 dump 带 vx vy vz 列;参考峰 O-H ~108 THz / 3600 cm-1)
& $py <skill-dir>/scripts/l3_vacf_vdos.py --dump run2_uw.dump --data data.lammps `
    --elements "O H"

# 9. 界面水驻留时间 + 偶极弛豫(先定义界面层区域)
& $py <skill-dir>/scripts/l3_tcf.py --dump run2_uw.dump --data data.lammps `
    --mode both --region layer --layer top --trunc 3.0

# 10. 界面综合:偶极取向 + K 离子 2D 分布 + 水层性质
& $py <skill-dir>/scripts/l4_interface.py --dump run2_uw.dump --data data.lammps `
    --mode all --adsorbate-element K
```

单条命令模板可查各脚本 `--help`。通用参数:全部模块支持 `--region box|layer|sphere`(+ 各自参数,见 `region.py --help`)、`--outdir`、`--name`、`--elements-map`。

## 输出约定

- 每模块输出 CSV + PNG + 一行 `CONCLUSION:` 机器可读结论
- **AI 必须把 `CONCLUSION:` 行原样转述给用户**,不得改写数值
- CSV 列名即语义(`r(A),g(r),CN(r)` / `t_ps,all_msd_3d,...` / `z(A),rho` / `frame,n_hbonds`),可直接导入 Origin
- `*_uw.dump` 是原生 lammpstrj(`id type xu yu zu`),任何 LAMMPS 生态工具可读

## 下游衔接

分析产出可直接转接其他 MCP,无需格式转换:

| 需求 | 转接 |
|---|---|
| 结构渲染图(快照/等值面/CNA 着色/粒子追踪) | **ovito-mcp** 直接打开 `run2_uw.dump`(原生 lammpstrj,零转换) |
| 发表级曲线(期刊样式) | **origin-mcp**:`origin_import_table(path="<out>/O-O_rdf.csv")` → 选列绘图 |
| 氢键网络可视化 | ovito-mcp 加载 dump + `--data data.lammps` 拓扑 |

## 已知坑(实测)

| 坑 | 现象 | 对策 |
|---|---|---|
| 三斜晶胞 dump | 解析报错 | MVP 不支持;正交化或文档标注;见 references |
| 缺 data.lammps/Masses | RDF/氢键无法选元素 | `--elements-map`;仅 MSD/密度可按 type 跑 |
| id 跳号 | 按行号取值错位 | 解析层存 id 列;`sort id` 输出 |
| MSD 未先 unwrap | D 虚高/平台 | 先 l1;`--unwrap pbc`(对已 unwrap 输入幂等) |
| COM 漂移 | MSD 超线性,D 虚高 | `--unwrap com --com-ref "1 2"`(慢扩散层参考) |
| 密度 mass 模式无元素映射 | 全部质量为 0 → 报错 | 传 --data / --elements-map |
| 水层误检(噪声纹波) | 层数偏多 | 层检测有显著极小阈值;`--smooth` 调大 |
| 氢键键表索引 | donor/acceptor 是 0-based index 非 LAMMPS id | 内部已统一;区域过滤按索引 |
| 帧间行序可变(未 sort id) | 按行号取原子错位 | 序参量/分子识别按 LAMMPS id 定位;建议 dump `sort id` |
| dump 无速度列 | VDOS 报错 | LAMMPS dump 加 `vx vy vz` 列 |
| 静态驻留 C(t)≠1 陷阱 | 误以为实现 bug | 键寿命是 (L−t)/L 衰减;residence TCF(Impey)才恒 1 |
| xu/yu/zu image 漂移 | HBA 氢键数伪下降(实测 160→100,真实恒定) | l2_hbonds 自动配对检查会 WARNING;用 wrapped dump(mod 回盒)重跑;MIC 类分析不受影响;详见 method_notes §3b |
| 邻居搜索含异种原子 | 序参量被 H 污染 | l2_order 只取同元素第一壳层 |
| MDAnalysis 版本 | 旧 hbonds 路径 3.0 移除;scipy 旧 sph_harm 移除 | 已用 `hydrogenbonds` / `sph_harm_y`;method_notes 记录实测版 |

## 权限与安全

- **本地计算**:脚本只读写 `--outdir` 与输入文件,不触碰集群(拉取走 cluster-connect)
- `bootstrap_env.py --force` 会删除 `.venv`,执行前确认
- 开源仓库不含真实路径/凭据;`meta.json` 中的 `source_dump` 是本地运行产物,不入仓库

## Reference Files

| 需要 | 读 |
|---|---|
| dump 格式、列名语义、三斜限制 | `references/lammps_dump_format.md` |
| 单位链、氢键判据、区域语义、版本记录 | `references/method_notes.md` |
| MD-Analyzer 移植来源与规避项 | `references/md_analyzer_notes.md` |
| 解析层 API(unwrap/write dump) | `scripts/dump_io.py` |
| 区域参数 | `scripts/region.py`(运行 `--help`) |

## Acknowledgements

- `scripts/l3_msd.py` 算法源自本项目 `calc_msd.py`(VASP AIMD MSD 分析,本仓库移植并参数化)
- 区域选择与 C(t)/寿命拟合移植自 [Maygon0303/MD-Analyzer](https://github.com/Maygon0303/MD-Analyzer)(MIT),详见 `references/md_analyzer_notes.md`
- 模板结构参考本项目 [vaspkit-remote](../vaspkit-remote/SKILL.md) 与 [cluster-connect](../cluster-connect/SKILL.md)
