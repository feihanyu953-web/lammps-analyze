# lammps-analyze

**LAMMPS / MACE-MD 轨迹本地后处理 skill for Claude Code。** 「远程拉取 + 本地分析」：轨迹经 [cluster-connect](https://github.com/feihanyu953-web/cluster-connect) 拉回本机后全部本地计算，零集群负载。输入 `dump.lammpstrj` + `data.lammps`，输出 **CSV + PNG + 一行机器可读结论（`CONCLUSION:`）**。

与 [vaspkit-remote](https://github.com/feihanyu953-web/vaspkit-remote) 的分工：vaspkit-remote 是**远程执行**（集群上跑 VASPKIT），本 skill 是**远程拉取 + 本地分析**——MdAnalysis 生态的精细分析（氢键寿命、序参量、驻留时间等）在本地做更灵活。

## 分析能力总览

| 模块 | 脚本 | 分析内容 | 关键参数 |
|---|---|---|---|
| log 平衡检查 | `l1_log_parse.py` | 温度/能量水平、平衡判定 | — |
| 轨迹预处理 | `l1_traj_prep.py` | unwrap / COM 去漂移 / 重采样 / 区域过滤 | `--stride`、`--name`，输出 `*_uw.dump` + `*_meta.json` |
| RDF + 配位数 | `l2_rdf_cn.py` | g(r) 与 CN(r)，任意元素对 | `--pairs "O:O O:H"`、`--cn-cut 3.5` |
| 密度分布 / 水层识别 | `l2_density.py` | 沿轴密度分布（质量/数密度）、界面水层数 | `--axis z`、`--density-type mass`、`--layers`、双表面自动检测 `--layer-auto` |
| 氢键统计 + 寿命 | `l2_hbonds.py` | 氢键数目、连续/间歇寿命 C(t) | 水体系默认 3.5 Å / 150°；`--lifetime-method double` |
| 结构序参量 | `l2_order.py` | q4/q6、四面体序参量、LSI（冰水区分） | 只取同元素第一壳层，防 H 污染 |
| MSD + 扩散系数 | `l3_msd.py` | MSD、D（含扩散态检查） | `--elements "O K"`、`--dims all`、`--diffusive-check`；meta.json 自动提供 dt |
| VACF → VDOS | `l3_vacf_vdos.py` | 振动谱（参考峰 O–H ~108 THz / 3600 cm⁻¹） | 需 dump 带 vx vy vz |
| 界面水动力学 | `l3_tcf.py` | 驻留时间 / 偶极弛豫 | `--mode both --region layer --layer top --trunc 3.0` |
| 界面综合分析 | `l4_interface.py` | 偶极取向 + 吸附质 2D 分布 + 水层性质 | `--mode all --adsorbate-element K` |

通用参数：所有模块支持 `--region box|layer|sphere`、`--outdir`、`--name`、`--elements-map`。每个脚本 `--help` 有完整说明。

## 输入 / 输出契约

**输入**
- `dump.lammpstrj`（必需）；`data.lammps`（推荐，提供类型→元素映射；缺失时用 `--elements-map "1:O 2:H ..."` 兜底）；`log.lammps`（log 检查用）

**输出（每个分析模块统一）**
- CSV —— 列名即语义，可直接导入 Origin
- matplotlib PNG —— 快速查看
- `CONCLUSION:` 一行科学结论 —— 机器可读，AI 转述时必须原样保留不得改写
- `*_uw.dump` —— 标准 lammpstrj 格式（`id type xu yu zu`），OVITO / MDAnalysis 零转换直读

## 快速开始

```powershell
$py = "$HOME\.claude\skills\lammps-analyze\.venv\Scripts\python.exe"
$sk = "$HOME\.claude\skills\lammps-analyze"

# 1. 环境引导（首次，幂等）
& $py $sk/scripts/bootstrap_env.py

# 2. 从集群拉轨迹（可选，需 cluster-connect）
python $HOME\.claude\skills\cluster-connect\scripts\cluster.py pull <profile> run2/dump.lammpstrj ./

# 3. 预处理（unwrap + 重采样）
& $py $sk/scripts/l1_traj_prep.py --dump dump.lammpstrj --data data.lammps --stride 10 --name run2

# 4. 按需分析
& $py $sk/scripts/l2_rdf_cn.py    --dump run2_uw.dump --data data.lammps --pairs "O:O O:H"
& $py $sk/scripts/l2_density.py   --dump run2_uw.dump --data data.lammps --axis z --layers
& $py $sk/scripts/l2_hbonds.py    --dump run2_uw.dump --data data.lammps
& $py $sk/scripts/l3_msd.py       --dump run2_uw.dump --data data.lammps --elements "O K" --diffusive-check
& $py $sk/scripts/l2_order.py     --dump run2_uw.dump --data data.lammps --elements O --lsi
& $py $sk/scripts/l3_vacf_vdos.py --dump run2_uw.dump --data data.lammps --elements "O H"   # 需 vx vy vz
& $py $sk/scripts/l3_tcf.py       --dump run2_uw.dump --data data.lammps --mode both --region layer --layer top --trunc 3.0
& $py $sk/scripts/l4_interface.py --dump run2_uw.dump --data data.lammps --mode all --adsorbate-element K
```

## 安装

```powershell
# 用户级（所有项目可用）
Copy-Item -Recurse lammps-analyze $HOME\.claude\skills\lammps-analyze
# 或项目级
Copy-Item -Recurse lammps-analyze <project>\.claude\skills\lammps-analyze
```

## 架构特点

- **分层设计**：解析层纯 numpy（`dump_io.py`，无第三方依赖），分析层 numpy + MDAnalysis（成熟算法）
- **独立环境**：首次调用自动用 uv 建 `<skill-dir>/.venv`（Python 3.11），不污染其他 Python 环境，幂等
- **统一区域选择**：`--region box|sphere|layer`，含双表面自动检测（`--layer-auto`）
- **可转接下游**：结构渲染 → [ovito-mcp](https://github.com/)；发表级曲线 → origin-mcp（CSV 直接导入）
- **实测坑固化**：SKILL.md「已知坑」表收录 14 条实测坑（id 跳号、COM 漂移、xu/yu/zu image 漂移致氢键数伪降、MDAnalysis 3.0 API 移除等），AI 生成分析方案时自动规避

## 前置依赖

| 依赖 | 用途 | 安装 |
|---|---|---|
| uv | 环境引导（自动下载 Python 3.11） | `pip install uv` 或 `winget install astral-sh.uv` |
| cluster-connect | 从集群 pull 轨迹（可选） | 姊妹 skill |
| ovito-mcp / origin-mcp | 结构渲染 / 发表级出图（可选） | MCP 插件 |

numpy / matplotlib / MDAnalysis 由 `bootstrap_env.py` 自动装入独立 `.venv`。

## 测试

```powershell
& $py -m pytest $sk/tests -q
```

合成数据优先（随机行走已知 D 恢复 ±10%、格点 RDF 峰位、双水分子氢键），少量手写 fixture 在 `tests/fixtures/`。

## 已知限制

- 三斜晶胞 dump 不支持（MVP 明确报错）
- `--per-frame` 区域仅密度支持；RDF/MSD 为静态第一帧（MSD 物理必需）
- 全内存解析；超大轨迹（百万原子 × 万帧）请 `--stride` 降采样

## 生态位

| 环节 | 工具 |
|---|---|
| 建模（slab / 异质结 / 电解质盒） | [emcbuild](https://github.com/feihanyu953-web/emcbuild) |
| 输入文件生成与校验 | [input_build](https://github.com/feihanyu953-web/input_build) |
| 集群提交 / 传输 | [cluster-connect](https://github.com/feihanyu953-web/cluster-connect) |
| 集群侧 VASP 后处理 | [vaspkit-remote](https://github.com/feihanyu953-web/vaspkit-remote) |
| **MD 轨迹本地分析** | **本项目** |

各工具均可独立使用，仅上述集成点需要对应姊妹 skill。

## License

MIT — 见 [LICENSE](LICENSE)。

## Acknowledgements

- MSD 算法源自本项目 `calc_msd.py`（VASP AIMD 分析）
- 区域选择与 C(t)/寿命算法移植自 [Maygon0303/MD-Analyzer](https://github.com/Maygon0303/MD-Analyzer)（MIT，见 [references/md_analyzer_notes.md](references/md_analyzer_notes.md)）
