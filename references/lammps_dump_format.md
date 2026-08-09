# LAMMPS dump 格式速查

lammps-analyze 的解析层 (`scripts/dump_io.py`) 按 **列名驱动** 解析 dump,不假设列位置。

## 标准 lammpstrj 结构

```
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
451
ITEM: BOX BOUNDS pp pp pp        <- 正交:3 行数值
0 16.6312
0 16.6312
0 48.381463
ITEM: ATOMS id type x y z        <- 列名驱动,顺序无关
1 1 1.38593 1.38593 10.0
...
```

三斜(4 行 BOX BOUNDS,含 tilt):
```
ITEM: BOX BOUNDS xy xz yz pp pp pp
0 10 1.0
0 10 0.5
0 10 0.0
0.1 0.2 0.3        <- 第 4 行 tilt
```
**MVP 不支持三斜**,解析时报错(见 SKILL.md 已知坑)。

## 坐标列语义(wrapped 检测)

| 列名 | 含义 | `Dump.wrapped` |
|---|---|---|
| `x y z` | PBC 包裹坐标(在盒内) | `True` → 分析前需 unwrap |
| `xu yu zu` | 去包裹坐标(绝对位置) | `False` → unwrap 是 no-op |

MACE-LAMMPS 常用 `dump ... id type xu yu zu` 输出,已是 unwrapped。

## 解析器行为(MVP)

- 必须含 `id` 与 `type` 列,否则报错
- `id` 列**不假设连续**(实测 MACE 轨迹 id 跳号:1,2,3,7,8,9,13,14)
- `start/stop/stride` 在读取时流式应用
- `dt_fs` 推算:帧间 TIMESTEP 差的中位数 × MD 步长(fs);`--timestep-fs` 默认 0.5
- 大数据量:全内存读取;百万原子×万帧请用 `--stride` 降采样(README 有内存估算)

## 推荐的 dump 输出组合(LAMMPS input 侧)

```lammps
dump traj all custom ${DUMP_EVERY} dump.lammpstrj id type xu yu zu
dump_modify traj sort id
thermo_style custom step temp press pe ke
```

`sort id` 保证同一原子在帧间行序一致(解析层也按 id 列取值,双保险)。

## 与 data.lammps 的配合

- 类型→元素映射优先级:`--elements-map "1:O 2:H"` > data.lammps `Masses` 段 `# 元素` 注释 > 无映射(仅 MSD/密度可按 type 分析)
- data 文件 `Atoms # full` 段列序:`ID mol type x y z`(测试里踩过 type 列错位的坑)
- MDAnalysis 加载:拓扑用 `format="DATA"`,轨迹必须显式 `format="LAMMPSDUMP"`(否则 DCD 默认读取器误读)
