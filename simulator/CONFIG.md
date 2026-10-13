# SILF Simulator 配置说明

周期仿真器，不渲染图像。给定一份 yaml，它回答一帧要多少 ASIC 周期、各模块占多久、DRAM 上走了多少字节、等效 FPS 是多少。时间单位是 ASIC 周期。`clock_frequency: 1.0` 时 1 周期 = 1 ns。

依赖：`simpy`、`pyyaml`。只有 `workload.use_synthetic: false` 且要加载真实模型时才需要本仓库的 PyTorch / CUDA rasterizer。

## 运行

在 LangSplatV2 仓库根目录执行：

```bash
python -m simulator.main --config simulator/configs/silf.yaml
```

不写 `--config` 时默认就是 `simulator/configs/silf.yaml`。

论文里的 DRAM traffic 对比（HLS、SCAR 关闭，九个场景各用自己的 checkpoint，测试集全部相机）用：

```bash
bash scripts/run_dram_traffic.sh
```

`GPU=1 bash scripts/run_dram_traffic.sh` 可以换卡。脚本会激活 `langsplat_v2` 环境，写出 `simulator/results/dram_traffic_summary.json` 和同名的 `.csv`。

同一套真实负载上的整体消融（ASIC / PIM，各自再开 HLS、HLS+SCAR）用：

```bash
bash scripts/run_ablation.sh
```

结果在 `simulator/results/ablation_summary.json` 和同名的 `.csv`。加速比相对「ASIC、HLS 和 SCAR 都关」的全帧。

四份配置只改对比开关，其余硬件参数相同：

| 文件 | `semantic.placement` | HLS | SCAR | 用途 |
| --- | --- | --- | --- | --- |
| `configs/silf.yaml` | `pim` | 开 | 开 | 完整 SILF |
| `configs/asic_only.yaml` | `asic` | 开 | 开 | 同样算法，SPU 挪到 ASIC，特征要读回 |
| `configs/fullframe_pim.yaml` | `pim` | 关 | 关 | 只看 PIM，不做层级剪枝 |
| `configs/fullframe_asic.yaml` | `asic` | 关 | 关 | 上面这一组的 ASIC 对照 |

结果写到 `output.stats_file`。终端会打印周期、FPS、像素数和分项流量。

多场景：把 `simulation.scene` 留空（`""`），并设置 `simulation.dataset`。`lerf` 会跑 figurines、teatime、ramen、kitchen；`3d_ovs` 会跑 bed、lawn、room、sofa、bench。各场景的分辨率和 ROI / fallback / SCAR 跳过率来自论文表，写进同一个 JSON，并给出 `average_fps`。

## 一帧怎么走

1. **GFP**：剔除、投影、把高斯分到 tile。
2. **HSE → URE → SDE**：tile 之间流水。排序、光栅、码本解码，并把 512 维特征写入 DRAM。
3. **语义阶段**：等这一级的特征图全部写完才开始。特征图只解码、只写入 DRAM 一次，和 query 数无关。之后每条 query 都要对这张图做一次 \(s=f\cdot q\)（\(n=f\cdot f\) 跟这次扫描一起做，不另开一趟）。ASIC 每条 query 都把整张特征图从 DRAM 读回再计算。PIM 的特征留在 bank 的 row buffer 里，query 再多也不把 \(f\) 读回 ASIC，只按 query 数回传 \((s,n)\) 或谓词。一行在所有 query 算完前保持打开。
4. 开了 HLS 时，粗阶段做完语义计算后，**FTG** 生成细任务。细阶段不再投影、不再排序，只光栅和解码 ROI 里还要细化的像素，然后再做一次语义阶段。

因此端到端时间大约是

```
T_pim  ≈ T_gfp + T_sde + T_spu
T_asic ≈ T_pim + T_reload
```

\(T_{reload}\) 只有 ASIC 有。排序和光栅通常叠在解码下面，不另加一截。多帧是一帧做完再做下一帧，FPS 是单帧延迟的倒数，不是多帧吞吐。

打印里的 `per_pixel_cycles` 是单个稠密像素的孤立代价，方便对照上面的式子：

```
sde             = num_levels * decode_k * clip_dim / macs_per_cycle
feature_write   = feature_bytes / (memory_bandwidth / clock_frequency)
feature_readback = feature_write * num_queries    仅 placement=asic，否则为 0
spu_stream      = ceil(clip_dim / lanes) * (clock_frequency / pim_frequency) / num_spus * num_queries
```

`spu_stream` 不含行打开和 refresh，所以会略小于 JSON 里的 `semantic_cycles / 像素数`。

## simulation

| 键 | 含义 |
| --- | --- |
| `dataset` | `lerf` 或 `3d_ovs`。`scene` 为空时用来枚举场景。 |
| `scene` | 场景名，如 `figurines`。空字符串表示扫整个 dataset。未知名字不会套论文表。 |
| `num_frames` | 合成负载时，把同一帧重复几次。帧之间不重叠。真实模型按 `camera_ids` 的相机数来，不再乘这个数。 |
| `resolution` | `宽x高`，如 `986x728`。省略时用场景表：LERF `986x728`，3D-OVS `1440x1080`。真实模型以相机分辨率为准。 |
| `model_path` | 权重目录。只在 `use_synthetic: false` 时使用。 |
| `source_path` | 场景数据目录，里面要有 `sparse/` 或 `transforms_train.json`。 |
| `iteration` | 传给 `Scene(..., load_iteration=)`。`-1` 表示最新一次迭代。 |
| `camera_split` | `test` 或 `train`。 |
| `camera_ids` | 用哪些相机，如 `0`、`"0,1,2"`，或 `all`（该 split 的全部相机）。越界的下标跳过。 |
| `num_queries` | 一帧特征图要匹配的文本 query 数，可以是小数（标注帧平均）。省略时用场景表。未知场景默认 1。yaml 里写了就覆盖场景表。 |
| `feature_level` | 加载真实模型时的特征层，默认 1。 |

`scene` 能对上、且 yaml 没写分辨率或 ROI / fallback / SCAR 时，自动填上。分辨率是 LangSplatV2 在 `resolution=-1` 下的实际推理尺寸；ROI、fallback、SCAR 仍是论文里的查询平均：

| 场景 | 分辨率 | query | ROI | fallback | SCAR 跳过 |
| --- | --- | --- | --- | --- | --- |
| figurines | 986×728 | 14 | 0.347 | 0.11 | 0.68 |
| teatime | 986×728 | 9.833 | 0.400 | 0.08 | 0.59 |
| ramen | 986×728 | 10.143 | 0.582 | 0.07 | 0.55 |
| kitchen | 986×728 | 4.4 | 0.630 | 0.13 | 0.61 |
| bed | 1440×1080 | 6 | 0.642 | 0.15 | 0.65 |
| lawn | 1440×1080 | 6 | 0.552 | 0.16 | 0.52 |
| room | 1440×1080 | 6 | 0.414 | 0.06 | 0.68 |
| sofa | 1440×1080 | 6 | 0.523 | 0.10 | 0.61 |
| bench | 1440×1080 | 7 | 0.562 | 0.12 | 0.72 |

论文表里的 ROI 已经包含 tile 膨胀，仿真不会再膨胀一次。

## hardware

| 键 | 含义 |
| --- | --- |
| `clock_frequency` | ASIC 时钟，GHz。GFP、HSE、URE、SDE、FTG 都用它。1.0 表示 1 周期 = 1 ns。 |
| `pim_frequency` | SPU / PDU 时钟，GHz。论文是 0.2。ASIC-only 也用这个频率，不升到 1 GHz。 |

### gfp

高斯前端，一帧一次。

| 键 | 含义 |
| --- | --- |
| `cull_cycles` | 每个场景高斯的剔除周期。场景高斯数 = `ceil(可见高斯数 / culling_survival_rate)`。 |
| `project_cycles` | 每个可见高斯的投影周期。 |
| `bin_cycles` | 每个可见高斯的 tile 分配周期。 |
| `culling_survival_rate` | 可见高斯占场景高斯的比例，范围 (0, 1]。为 1 时前端只读可见高斯。 |

前端读入的字节数是 `场景高斯数 × memory.geom_bytes`。

### hse

GSCore 式层次排序。一个核负责一个 tile 的全部 chunk。细阶段在 HLS 打开时跳过排序。

| 键 | 含义 |
| --- | --- |
| `num_cores` | 排序核个数。默认 4，对应 4 个 ABU。 |
| `coarse_sort_cycles` | 每个 chunk 的粗排周期，不随 chunk 里的高斯数变化。 |
| `fine_sort_cycles` | 每个 chunk 的细排周期。一个 chunk 的排序 = 两者之和。 |
| `bytes_per_gaussian` | 排序键的字节数（深度 + id）。只对没和前一个 tile 重叠的高斯计数。 |

chunk 个数 = `ceil(tile 内高斯数 / algorithm.chunk_size)`。chunk 切得更碎，排序总周期线性增加。

### ure

统一光栅。4 个 ABU，每个 tile 占一个。RGB 和语言码字权重一起 blend。

| 键 | 含义 |
| --- | --- |
| `num_abus` | 并行的 Alpha Blending Unit 个数。 |
| `cores_per_abu` | 每个 ABU 的光栅核数。`4 × 64 = 256`，对应 16×16 tile。 |
| `cycles_per_pass` | 一组核完成一次 blend 的周期。 |
| `bytes_per_gaussian` | 光栅时每个高斯读的字节。`0` 表示 `geom_bytes + 语言载荷`。 |

一个 tile 的光栅周期：

```
有效像素 / cores_per_abu * 高斯数 * early_stop_ratio * cycles_per_pass
```

`early_stop_ratio` 在 `algorithm` 里，表示 alpha 饱和后只处理一部分高斯。语言载荷：`sparse_decode` 时每层 `decode_k` 个 FP16 权重加 uint16 下标（每层 `decode_k × 4` 字节）；否则每层 `codebook_size × 2` 字节。

### sde

码本解码，一个脉动阵列。码本留在片上。

| 键 | 含义 |
| --- | --- |
| `array_dim` | 论文表里的 K。`macs_per_cycle` 为 0 时，每周期 MAC 数 = `array_dim²`。 |
| `macs_per_cycle` | 每周期 MAC 数。大于 0 时覆盖 `array_dim²`。 |
| `decode_k` | 每个像素、每一层实际累加的码字数。`sparse_decode: true` 时用它。 |
| `codebook_size` | 码本宽度。`sparse_decode: false` 时每个像素做完整的 `codebook_size × clip_dim` 乘加。 |
| `num_levels` | 特征层数。LangSplatV2 评测把三层拼在一起，默认 3。 |
| `clip_dim` | 解码后的 CLIP 维度，默认 512。 |
| `sparse_decode` | true：只累加 `decode_k` 个码字。false：稠密码本。 |
| `output_buffer_tiles` | 片上特征缓冲能压住几个 tile。缓冲满时解码停住，直到该 tile 的向量写入 DRAM。 |

每像素解码周期 = `num_levels × 码字数 × clip_dim / macs_per_cycle`。默认 `3 × 8 × 512 / 614 ≈ 20`，和一次 1024 字节的 DRAM 传输同量级。若改回 `macs_per_cycle: 0`（即 64 MAC/周期），解码变成 192 周期/像素，PIM 省下的读回几乎不体现在 FPS 里。

### ftg

粗阶段语义结果出来之后才运行。`hls_enabled: false` 时不跑，周期为 0。

| 键 | 含义 |
| --- | --- |
| `num_vrus` | 并行 VRU 个数，论文是 4。 |
| `cycles_per_coarse_pixel` | 每个粗像素做 relevance 和 SCAR 判断的周期，由 VRU 平分。 |
| `dilation_cycles_per_tile` | 每个粗 tile 做膨胀 / 任务入队的周期，同样由 VRU 平分。 |

ROI 的面积用的是论文统计，不由这里的膨胀周期改变。

### semantic

归一化和查询匹配。`placement` 是 PIM 和 ASIC-only 的唯一功能差别。

| 键 | 含义 |
| --- | --- |
| `placement` | `pim` 或 `asic`。 |
| `num_channels` | DRAM 通道数，论文是 4。每通道一个 PDU。 |
| `banks_per_channel` | 每通道 bank 数，论文是 16。 |
| `spus_per_channel` | 每通道 SPU 数。8 表示两个 bank 共用一个 SPU，总共 32 个。 |
| `lanes` | 每个 SPU 的 FP16 通道数。\(s=f\cdot q\) 和 \(n=f\cdot f\) 同时算，所以是 16 个乘法器，但向量扫描周期按 `lanes` 算。 |
| `feature_bytes` | 每个像素写入 DRAM 的特征字节。512 × FP16 = 1024。ASIC 读回同样这么多。 |
| `result_bytes_coarse` | PIM 粗阶段回传字节。两个 FP16 的 \((s, n)\) = 4。 |
| `result_bytes_fine` | PIM 细阶段回传字节。1 bit 谓词 = 0.125。 |
| `pdu_cycles_per_pixel` | 细阶段 PDU 的周期，单位是 PIM 周期，再换算成 ASIC 周期。 |

SPU 周期按像素条带到 32 个 SPU 上。每一行支付 `tRCD + tRP`，同一通道上多个 SPU 的 ACT 相隔 `tRRD`，并加 `tRFC / tREFI` 的 refresh。ASIC 在此之前还要整图读回 `feature_bytes`；PIM 只在算完后回传 `result_bytes_*`。

### memory

片外总线，峰值带宽由四通道共享。几何读可以隐藏一部分，特征写和读回不隐藏。

| 键 | 含义 |
| --- | --- |
| `memory_bandwidth` | 聚合带宽，GB/s。论文 51.2 = 4 × x16 × 6.4 Gb/s/pin。 |
| `geom_bytes` | 每个高斯的几何字节（坐标、尺度、旋转、不透明度、颜色一类）。不含语言码字。 |
| `read_latency_hiding_rate` | 几何读（前端、排序、光栅）可与计算重叠的比例。0.3 表示只露出 70% 的传输时间。特征写、特征读回、结果回传不用这个系数。 |
| `row_buffer_bytes` | 每个 bank 的 row buffer，论文 2 KB。一行能放 `row_buffer_bytes / feature_bytes` 个像素向量。 |
| `tRCD_ns` | 行激活，论文 18 ns。1 GHz 下就是 18 周期。 |
| `tRP_ns` | precharge，论文 18 ns。 |
| `tRAS_ns` | 最小行打开时间，论文 42 ns。SPU 扫描一行若更长，取扫描时间。 |
| `tWR_ns` | 写恢复，论文 34 ns。只加在特征写上。 |
| `tRRD_ns` | 同一通道上两次 ACT 的间隔。论文表没列，按 LPDDR5 取 5 ns，用来表示 bank group 约束。 |
| `tREFI_ns` | 平均刷新间隔，3906 ns。 |
| `tRFC_ns` | 一次刷新占用，180 ns。按 `tRFC / tREFI` 摊进传输和 SPU 时间。 |

特征传输周期取「字节 / 每周期字节数」和「行命令时间」里的较大者，再加 refresh。

## algorithm

| 键 | 含义 |
| --- | --- |
| `tile_size` | tile 边长，像素。论文 16。边缘 tile 可以不满。 |
| `chunk_size` | 排序 chunk 的高斯数。 |
| `coarse_downscale` | 粗渲染相对全分辨率的边长倍数。4 表示宽和高都除以 4。 |
| `hls_enabled` | 是否做粗定位再细分割。关闭时没有粗阶段、没有 FTG，细阶段是全帧并且要重新排序。 |
| `scar_enabled` | 是否在 ROI 内跳过语义稳定像素。HLS 关闭时此项无效。 |
| `early_stop_ratio` | 光栅里实际 blend 的高斯比例。 |
| `roi_fraction` | 细阶段 ROI 占画面的比例。省略则用论文表。ROI 取画面中心的一块矩形。 |
| `fallback_rate` | 粗阶段失败、退回全帧细渲染的查询比例。非 ROI tile 按这个比例抽一部分整 tile 渲染；ROI 内的像素比例是 `(1 - fallback) × (1 - scar_skip) + fallback`。 |
| `scar_skip_ratio` | ROI 内被 SCAR 跳过、不再做细渲染的像素比例。 |

## workload

| 键 | 含义 |
| --- | --- |
| `use_synthetic` | true：按下面两个参数铺高斯，不加载模型。false：用该场景自己的 checkpoint 投影。加载失败直接报错，不再退回合成场。 |
| `num_gaussians` | 合成负载的可见高斯数。 |
| `gaussian_tile_cover` | 每个高斯覆盖的 tile 边长。2 表示盖住 2×2 个 tile，用来产生相邻 tile 的高斯重叠。 |

合成布局是确定性的：第 `i` 个高斯落在 `(i*17 % ntx, i*13 % nty)`。多场景扫描时若各场景共用同一个 `num_gaussians`，帧率差别只来自 ROI / SCAR，不来自几何。

`use_synthetic: false` 且 yaml 没写路径时，按场景名自动找权重：LERF 用 `output/<场景>_0_1` 和 `/data/home/rh_xu30/data/lerf_ovs/<场景>`，3D-OVS 同样。论文里的 kitchen 对应磁盘上的 `waldo_kitchen`。`output/bench_0_1` 没有语言点云，bench 改用 `output/bench_rgb` 的 RGB 3DGS。训练时写进 `cfg_args` 的 `eval` 和 `resolution` 不沿用；推理固定 `resolution=-1`、测试集（每 8 张取 1 张）。`camera_ids: all` 会把这些测试视角都跑完，JSON 里的 `dram_bytes` 是各帧之和，`fps` 已经是按帧平均。仿真用 `radii > 0` 的高斯，屏幕坐标由 `full_proj_transform` 自己投影（`render()` 交回的 `means2D` 是全 0）。同一份 tile 列表会缓存，PIM 和 ASIC-only 不用各投影一次。

## output

| 键 | 含义 |
| --- | --- |
| `stats_file` | JSON 路径。多场景也写这个文件，内容带 `"mode": "multi_scene"`。 |
| `verbose` | 是否在终端打印摘要。 |

JSON 里和论文直接相关的字段：

| 字段 | 含义 |
| --- | --- |
| `total_cycles` | 仿真结束时的周期，即单帧（或若干串行帧）的墙钟延迟。 |
| `fps` | `clock_frequency × 1e9 / 平均每帧周期`。 |
| `gfp_cycles` / `sort_cycles` / `raster_cycles` / `decode_cycles` / `ftg_cycles` / `semantic_cycles` | 各模块有人在忙的时间。排序、光栅、解码是多核并集；语义阶段包含 ASIC 读回和 SPU。它们和 `total_cycles` 重叠，不能相加。 |
| `dram_cycles` | 片外总线被占用的时间。 |
| `dram_bytes` | 分项字节：`gaussian_read`、`sort_read`、`raster_read`、`feature_write`、`feature_readback`（仅 ASIC）、`result_return`（仅 PIM）。 |
| `pixels` | `full` 全帧像素，`coarse` / `fine` 实际解码的像素，以及用到的 ROI、fallback、SCAR 比例。 |
| `notes` | 上面的每像素孤立周期，外加 `gaussian_bytes`。 |
| `frame_times` | 每帧秒数。 |

`semantic_cycles` 在 ASIC 上应明显大于 PIM，差值接近 `feature_readback` 的传输周期。若 FPS 几乎不动，先看 `notes.sde` 是否远大于 `notes.feature_readback`。

## 输出转化csv

```
python -m simulator.export_csv simulator/results/dram_traffic_summary.json
```