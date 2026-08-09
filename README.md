# lammps-analyze

LAMMPS / MACE-MD 轨迹**本地**后处理 Claude Code skill:RDF/配位数、密度分布与界面水层识别、氢键统计与寿命、MSD 与扩散系数、log 平衡检查。输入 `dump.lammpstrj` + `data.lammps`,输出 CSV + PNG + 一行机器可读结论。

## 特性

- **本地分析**:轨迹经 cluster-connect 拉回本机后全本地计算,零集群负载
- **分层架构**:解析层纯 numpy(`dump_io.py`,无第三方依赖),分析层 numpy + MDAnalysis(成熟算法)
- **独立环境**:首次调用自动用 uv 建 `<skill-dir>/.venv`(Python 3.11),不污染其他 Python 环境,幂等
- **统一区域选择**:`--region box|sphere|layer`,含双表面自动检测(`--layer-auto`)
- **原生中间格式**:`*_uw.dump` 是标准 lammpstrj,OVITO / MDAnalysis / 下游工具零转换直读
- **可转接**:渲染级结构图 → ovito-mcp;发表级曲线 → origin-mcp(CSV 直接导入)
- **输出契约**:CSV + matplotlib PNG + `CONCLUSION:` 一行科学结论

## 前置依赖

| 依赖 | 用途 | 安装 |
|---|---|---|
| uv | 环境引导(自动下载 Python 3.11) | `pip install uv` 或 `winget install astral-sh.uv` |
| cluster-connect | 从集群 pull 轨迹 | 本项目姊妹 skill,可选 |

numpy / matplotlib / MDAnalysis 由 `bootstrap_env.py` 自动装入独立 `.venv`。

## 安装

```powershell
# 全局安装(所有项目可用)
Copy-Item -Recurse <repo-path> $HOME\.claude\skills\lammps-analyze

# 或项目级
Copy-Item -Recurse <repo-path> <project>\.claude\skills\lammps-analyze
```

## 快速开始

```powershell
$py = "$HOME\.claude\skills\lammps-analyze\.venv\Scripts\python.exe"
$sk = "$HOME\.claude\skills\lammps-analyze"

# 1. 环境引导(首次)
& $py $sk/scripts/bootstrap_env.py

# 2. 从集群拉轨迹
python $HOME\.claude\skills\cluster-connect\scripts\cluster.py pull <profile> run2/dump.lammpstrj ./

# 3. 预处理(unwrap + 重采样)
& $py $sk/scripts/l1_traj_prep.py --dump dump.lammpstrj --data data.lammps --stride 10 --name run2

# 4. 分析
& $py $sk/scripts/l2_rdf_cn.py   --dump run2_uw.dump --data data.lammps --pairs "O:O O:H"
& $py $sk/scripts/l2_density.py  --dump run2_uw.dump --data data.lammps --axis z --layers
& $py $sk/scripts/l2_hbonds.py   --dump run2_uw.dump --data data.lammps
& $py $sk/scripts/l3_msd.py      --dump run2_uw.dump --data data.lammps --elements "O K" --diffusive-check
```

每个脚本 `--help` 有完整参数说明。

## 测试

```powershell
& $py -m pytest $sk/tests -q
```

合成数据优先(随机行走已知 D 恢复 ±10%、格点 RDF 峰位、双水分子氢键),少量手写 fixture 在 `tests/fixtures/`。

## 已知限制

- 三斜晶胞 dump 不支持(MVP 明确报错)
- `--per-frame` 区域仅密度支持;RDF/MSD 为静态第一帧(MSD 物理必需)
- 全内存解析;超大轨迹(百万原子×万帧)请 `--stride` 降采样

## License

MIT — 见 [LICENSE](LICENSE)。

## Acknowledgements

- MSD 算法源自本项目 `calc_msd.py`(VASP AIMD 分析)
- 区域选择与 C(t)/寿命算法移植自 [Maygon0303/MD-Analyzer](https://github.com/Maygon0303/MD-Analyzer)(MIT,见 [references/md_analyzer_notes.md](references/md_analyzer_notes.md))
- 模板参考 [vaspkit-remote](https://github.com/feihanyu/vaspkit-remote) 与 [cluster-connect](https://github.com/feihanyu/cluster-connect)
