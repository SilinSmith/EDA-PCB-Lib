# 环境开关总表

> 本文件由 `python tools/dev/gen_knobs.py --write` **自动生成**，不要手改。

全部调参都走环境变量。假值词表（`0` / `off` / `false` / `none` / `no` 一律视为关）
统一实现在 `envflag.py`；PowerShell 里 `$env:X=''` 是**删除**变量而不是关功能，
要关某个开关请写具体假值，例如 `$env:STAGE2_PINNUM_TIGHT='0'`。

共 **664** 个开关。

## 目录

- [stage1](#stage1)（165）
- [stage2](#stage2)（245）
- [stage3](#stage3)（2）
- [fusion](#fusion)（133）
- [vlm](#vlm)（51）
- [textrec](#textrec)（13）
- [cvparts](#cvparts)（11）
- [pipeline](#pipeline)（22）
- [submission](#submission)（18）
- [schematic_parser](#schematic_parser)（4）

<a id="stage1"></a>

## stage1

| 开关 | 默认值 | 定义位置 |
|---|---|---|
| `BIGIC_FILL_MIN` | `"0.88"` | `stage1/bigic.py:47` |
| `BIGIC_INK` | `"140"` | `stage1/bigic.py:50` |
| `BIGIC_MAX_FRAC` | `"0.55"` | `stage1/bigic.py:46` |
| `BIGIC_MIN_AREA` | `"8000"` | `stage1/bigic.py:45` |
| `BIGIC_MIN_SIDE` | `"60"` | `stage1/bigic.py:44` |
| `BIGIC_REJECT_WIRE` | `"1"` | `stage1/bigic.py:73` |
| `BIGIC_STUB_BAND` | `"0.6"` | `stage1/bigic.py:51` |
| `BIGIC_STUB_MIN` | `"3"` | `stage1/bigic.py:48` |
| `BIGIC_STUB_OFF` | `"10"` | `stage1/bigic.py:49` |
| `BIGIC_WIRE_BAND` | `"2"` | `stage1/bigic.py:75` |
| `BIGIC_WIRE_FRAC` | `"0.5"` | `stage1/bigic.py:74` |
| `EDA_TEXTBOX_INK_WIDE` | `"0"` | `stage1/pipeline.py:549` |
| `LLM_API_KEY` | `""` | `stage1/llm.py:141` |
| `LLM_BASE_URL` | `""` *(4 处)* | `stage1/llm.py:81` |
| `LLM_BATCH` | `"8"` | `stage1/llm.py:50` |
| `LLM_MAX_TOKENS` | `"4096"` | `stage1/llm.py:52` |
| `LLM_MAX_TOKENS_CAP` | `"16384"` | `stage1/llm.py:53` |
| `LLM_MODEL` | — *(3 处)* | `stage1/llm.py:91` |
| `LLM_PROBE_TIMEOUT` | `"5"` | `stage1/llm.py:48` |
| `LLM_RETRY` | `"1"` | `stage1/llm.py:54` |
| `LLM_TEMPERATURE` | `"0.0"` | `stage1/llm.py:55` |
| `LLM_TIMEOUT` | `"180"` | `stage1/llm.py:45` |
| `LLM_WORKERS` | `"4"` | `stage1/llm.py:51` |
| `STAGE1_BIGIC` | `"1"` | `stage1/pipeline.py:79` |
| `STAGE1_BLANK_FILTER` | `"0"` | `stage1/pipeline.py:425` |
| `STAGE1_CROSS_COS` | `"-0.85"` | `stage1/labels.py:175` |
| `STAGE1_CROSS_DOT_NEED` | `"3"` | `stage1/labels.py:177` |
| `STAGE1_CROSS_DOT_R` | `"3.0"` | `stage1/labels.py:176` |
| `STAGE1_CROSS_GAP` | `"10"` | `stage1/labels.py:210` |
| `STAGE1_CROSS_SPLIT` | `"1"` | `stage1/labels.py:174` |
| `STAGE1_CROSS_TAGS` | `"1"` | `stage1/labels.py:179` |
| `STAGE1_CROSS_WIDE` | `"0"` | `stage1/labels.py:178` |
| `STAGE1_CROSS_WIDE_MAX` | `"96"` | `stage1/labels.py:180` |
| `STAGE1_CV_WIN` | `"off"` | `stage1/detect.py:134` |
| `STAGE1_DROP_TYPES` | `"pin,testpoint,mosfet_npn"` | `stage1/detect.py:97` |
| `STAGE1_DUAL_DETECT` | `"2560"` | `stage1/pipeline.py:451` |
| `STAGE1_DUAL_DETECT_DBG` | — | `stage1/pipeline.py:512` |
| `STAGE1_DUAL_FINE` | `"0.75"` | `stage1/pipeline.py:452` |
| `STAGE1_DUAL_MAX_PIXELS` | `"0"` | `stage1/pipeline.py:455` |
| `STAGE1_DUAL_MIN_SIDE` | `"1400"` | `stage1/pipeline.py:456` |
| `STAGE1_DUAL_NEW` | `"0.35"` | `stage1/pipeline.py:453` |
| `STAGE1_GROUP_TOL` | `"1.5"` | `stage1/labels.py:47` |
| `STAGE1_MAX_PIXELS_HARD` | — | `stage1/pipeline.py:853` |
| `STAGE1_MERGE_TP` | `"0"` | `stage1/detect.py:120` |
| `STAGE1_MONO_FRAME_PINS` | `"0"` | `stage1/pipeline.py:118` |
| `STAGE1_NMS_IOU` | `"0.5"` | `stage1/detect.py:81` |
| `STAGE1_OCC_BIG` | `"60"` | `stage1/pipeline.py:605` |
| `STAGE1_OCC_LOCAL` | `"0"` | `stage1/wires.py:70` |
| `STAGE1_OCC_MODE` | `"all"` | `stage1/pipeline.py:599` |
| `STAGE1_OWN` | `"off"` | `stage1/detect.py:73` |
| `STAGE1_OWN_DEMOTE` | `"0.85"` | `stage1/detect.py:75` |
| `STAGE1_PINC` | `"1"` | `stage1/pipeline.py:89` |
| `STAGE1_PINC_BAND` | `"22"` | `stage1/pincol.py:95` |
| `STAGE1_PINC_CENTRE_FRAC` | `"0.30"` | `stage1/pincol.py:880` |
| `STAGE1_PINC_CONNECT` | `"3"` | `stage1/pincol.py:101` |
| `STAGE1_PINC_COVER` | `"0.5"` | `stage1/pincol.py:122` |
| `STAGE1_PINC_FALLBACK` | `"0"` | `stage1/pincol.py:124` |
| `STAGE1_PINC_FILL` | `"1"` | `stage1/pincol.py:161` |
| `STAGE1_PINC_KEEP_REACH` | `"2.5"` | `stage1/pincol.py:116` |
| `STAGE1_PINC_MAX_GAP` | `"3"` | `stage1/pincol.py:120` |
| `STAGE1_PINC_MAX_SIDE` | `"200"` | `stage1/pincol.py:151` |
| `STAGE1_PINC_MERGE` | `"1"` | `stage1/pincol.py:165` |
| `STAGE1_PINC_MIN_EL` | `"4"` | `stage1/pincol.py:127` |
| `STAGE1_PINC_MIN_NUM` | `"2"` | `stage1/pincol.py:155` |
| `STAGE1_PINC_MIN_RUN` | `"1"` | `stage1/pincol.py:99` |
| `STAGE1_PINC_MIN_SIDE` | `"60"` | `stage1/pincol.py:153` |
| `STAGE1_PINC_MIN_STUB` | `"2"` | `stage1/pincol.py:97` |
| `STAGE1_PINC_NAME_REACH` | `"90"` | `stage1/pincol.py:132` |
| `STAGE1_PINC_NUM_MAXH` | `"26"` | `stage1/pincol.py:135` |
| `STAGE1_PINC_NUM_MAXW` | `"42"` | `stage1/pincol.py:134` |
| `STAGE1_PINC_NUM_REACH` | `"26"` | `stage1/pincol.py:131` |
| `STAGE1_PINC_OUT` | `"12.0"` | `stage1/pincol.py:129` |
| `STAGE1_PINC_PITCH_MAX` | `"48.0"` | `stage1/pincol.py:114` |
| `STAGE1_PINC_PITCH_MIN` | `"5.0"` | `stage1/pincol.py:113` |
| `STAGE1_PINC_SNAP` | `"5.0"` | `stage1/pincol.py:164` |
| `STAGE1_PINC_STUB_REACH` | `"0"` | `stage1/pincol.py:111` |
| `STAGE1_PINC_SUP_TOL` | `"0.5"` | `stage1/pincol.py:118` |
| `STAGE1_PINC_TIGHT_MINH` | `"3"` | `stage1/pincol.py:872` |
| `STAGE1_PINC_TIGHT_X` | `"2.0"` | `stage1/pincol.py:870` |
| `STAGE1_PINC_TIGHT_Y` | `"3.0"` | `stage1/pincol.py:871` |
| `STAGE1_PINC_TIP_PAD` | `"0"` | `stage1/pincol.py:148` |
| `STAGE1_PINC_TIP_TOL` | `"6.0"` | `stage1/pincol.py:149` |
| `STAGE1_PIN_ATTACHMENT_GUARD` | `"0"` | `stage1/pipeline.py:120` |
| `STAGE1_PIN_ATTR` | `"nearest"` | `stage1/nets.py:67` |
| `STAGE1_PIN_ATTR_SLACK` | `"6"` | `stage1/nets.py:68` |
| `STAGE1_PIN_AXIS` | `"1"` | `stage1/pins.py:257` |
| `STAGE1_PIN_AXIS_RATIO` | `"1.2"` | `stage1/pins.py:259` |
| `STAGE1_PIN_CLUSTER_IC` | `"12.0"` | `stage1/pins.py:111` |
| `STAGE1_PIN_COLOR` | `"0"` | `stage1/pipeline.py:101` |
| `STAGE1_PIN_COLOR_CAP` | `"0"` | `stage1/pipeline.py:106` |
| `STAGE1_PIN_COLOR_KEEP_MISS` | `"0"` | `stage1/pipeline.py:112` |
| `STAGE1_PIN_DIRECTION_GUARD` | `"0"` | `stage1/pipeline.py:119` |
| `STAGE1_PIN_INK_TEXT` | `"0"` | `stage1/pins.py:218` |
| `STAGE1_PIN_INK_TEXT_PAD` | `"0.0"` | `stage1/pins.py:220` |
| `STAGE1_PIN_JOIN_DIST` | `"12"` | `stage1/nets.py:37` |
| `STAGE1_PIN_JOIN_FAR` | `"0"` | `stage1/nets.py:49` |
| `STAGE1_PIN_JOIN_SYNTH` | `"3"` | `stage1/nets.py:59` |
| `STAGE1_PIN_LEAD` | `"26"` | `stage1/pins.py:293` |
| `STAGE1_PIN_LEAD_FREE` | `"1"` | `stage1/pins.py:300` |
| `STAGE1_PIN_LEAD_INK` | `"0"` | `stage1/pins.py:304` |
| `STAGE1_PIN_LEAD_INK_FRAC` | `"0.6"` | `stage1/pins.py:306` |
| `STAGE1_PIN_LEAD_ORPHAN` | `"0"` | `stage1/pins.py:311` |
| `STAGE1_PIN_LEAD_ORPHAN_D` | `"12.0"` | `stage1/pins.py:313` |
| `STAGE1_PIN_LEAD_PERP` | `"3.0"` | `stage1/pins.py:294` |
| `STAGE1_PIN_LEAD_SRC` | `"end"` | `stage1/pins.py:297` |
| `STAGE1_PIN_LEG_ALL` | `"0"` | `stage1/pins.py:72` |
| `STAGE1_PIN_LEG_BOX` | `"12,10,8,10"` | `stage1/pins.py:68` |
| `STAGE1_PIN_LEG_SNAP` | `"0"` | `stage1/pins.py:71` |
| `STAGE1_PIN_MAJOR` | `"1"` | `stage1/pins.py:273` |
| `STAGE1_PIN_OUT_BOX` | `"12.0"` | `stage1/pins.py:55` |
| `STAGE1_PIN_STUB` | `"gapfill"` | `stage1/pins.py:154` |
| `STAGE1_PIN_STUB_GAPTOL` | `"4.0"` | `stage1/pins.py:177` |
| `STAGE1_PIN_STUB_MINLEN` | `"8"` | `stage1/pins.py:176` |
| `STAGE1_PIN_STUB_RATIO` | `"0.6"` | `stage1/pins.py:158` |
| `STAGE1_PIN_STUB_TEXT` | `"1"` | `stage1/pins.py:206` |
| `STAGE1_PIN_STUB_TEXT_PAD` | `"0.0"` | `stage1/pins.py:208` |
| `STAGE1_SPLIT_JUNCTIONS` | `"1"` | `stage1/wires.py:917` |
| `STAGE1_SPLIT_JUNC_NODES` | `"1"` | `stage1/wires.py:928` |
| `STAGE1_SPLIT_JUNC_TOL` | `"2.5"` | `stage1/wires.py:930` |
| `STAGE1_SPLIT_TOL` | `"1.5"` | `stage1/wires.py:919` |
| `STAGE1_TEXT_BRIDGE` | `"0"` | `stage1/labels.py:110` |
| `STAGE1_TEXT_BRIDGE_PAD` | `"3.0"` | `stage1/labels.py:111` |
| `STAGE1_WIRE_BRIDGE` | `"0"` | `stage1/wires.py:893` |
| `STAGE1_WIRE_BRIDGE_ANGLE` | `"30"` | `stage1/wires.py:898` |
| `STAGE1_WIRE_BRIDGE_DE` | `"22"` | `stage1/wires.py:900` |
| `STAGE1_WIRE_BRIDGE_FRAC` | `"0.9"` | `stage1/wires.py:897` |
| `STAGE1_WIRE_BRIDGE_INK` | `"0.5"` | `stage1/wires.py:899` |
| `STAGE1_WIRE_BRIDGE_MIN_SIDE` | `"24"` | `stage1/wires.py:901` |
| `STAGE1_WIRE_BRIDGE_NEAR` | `"8"` | `stage1/wires.py:896` |
| `STAGE1_WIRE_BRIDGE_THICK` | `"9"` | `stage1/wires.py:902` |
| `STAGE1_WIRE_CLOSE_THIN` | `"0"` | `stage1/wires.py:659` |
| `STAGE1_WIRE_COLOUR` | `"confirm"` | `stage1/wires.py:551` |
| `STAGE1_WIRE_COS` | `"-0.75"` | `stage1/labels.py:144` |
| `STAGE1_WIRE_DILATE` | `"0"` | `stage1/wires.py:672` |
| `STAGE1_WIRE_EXT` | `"8.0"` | `stage1/labels.py:95` |
| `STAGE1_WIRE_EXT_NEAREST` | `"0"` | `stage1/labels.py:97` |
| `STAGE1_WIRE_EXT_PERP` | `"2.0"` | `stage1/labels.py:96` |
| `STAGE1_WIRE_HUE_MAXFAIL` | `"3"` | `stage1/wires.py:507` |
| `STAGE1_WIRE_HUE_MAXSIDE` | `"1024"` | `stage1/wires.py:525` |
| `STAGE1_WIRE_HUE_TIMEOUT` | `"30"` | `stage1/wires.py:506` |
| `STAGE1_WIRE_INK` | `"1"` | `stage1/labels.py:147` |
| `STAGE1_WIRE_INK_FRAC` | `"0.5"` | `stage1/labels.py:148` |
| `STAGE1_WIRE_INK_MODE` | `""` | `stage1/wires.py:668` |
| `STAGE1_WIRE_OVER` | `"12"` | `stage1/labels.py:235` |
| `STAGE1_WIRE_OVER_COS` | `"0.9"` | `stage1/labels.py:242` |
| `STAGE1_WIRE_OVER_CROSS` | `"0"` | `stage1/labels.py:241` |
| `STAGE1_WIRE_PERP` | `"1.5"` | `stage1/labels.py:143` |
| `STAGE1_WIRE_SNAP_GAPS` | `"1"` | `stage1/wires.py:939` |
| `STAGE1_WIRE_SNAP_GAPS_ANG` | `"25"` | `stage1/wires.py:944` |
| `STAGE1_WIRE_SNAP_GAPS_K` | `"4"` | `stage1/wires.py:945` |
| `STAGE1_WIRE_SNAP_GAPS_MAX` | `"18"` | `stage1/wires.py:943` |
| `STAGE1_WIRE_SNAP_GAPS_MODE` | `"short"` | `stage1/wires.py:946` |
| `STAGE1_WIRE_SNAP_GAPS_RATIO` | `"0.9"` | `stage1/wires.py:947` |
| `STAGE1_WIRE_SNAP_TEXT` | `"0"` | `stage1/wires.py:955` |
| `STAGE1_WIRE_SNAP_TEXT_ANG` | `"25"` | `stage1/wires.py:960` |
| `STAGE1_WIRE_SNAP_TEXT_COVER` | `"0.6"` | `stage1/wires.py:963` |
| `STAGE1_WIRE_SNAP_TEXT_K` | `"4"` | `stage1/wires.py:961` |
| `STAGE1_WIRE_SNAP_TEXT_LONG` | `"0"` | `stage1/wires.py:965` |
| `STAGE1_WIRE_SNAP_TEXT_MAX` | `"30"` | `stage1/wires.py:959` |
| `STAGE1_WIRE_SNAP_TEXT_PAD` | `"2"` | `stage1/wires.py:962` |
| `STAGE1_WIRE_SNAP_TEXT_SHORT` | `"0"` | `stage1/wires.py:964` |
| `STAGE1_WIRE_THIN_W` | `""` | `stage1/wires.py:656` |
| `STAGE1_WIRE_TOL` | `"1.5"` | `stage1/labels.py:94` |
| `STAGE1_WIRE_WELD` | `"16.0"` | `stage1/labels.py:142` |
| `VLM_TIMEOUT` | `"600"` | `stage1/wires.py:503` |

<a id="stage2"></a>

## stage2

| 开关 | 默认值 | 定义位置 |
|---|---|---|
| `CV_UNIFIED` | `"0"` | `stage2/run.py:542` |
| `EDA_TEXTBOX_INK_WIDE_MODE` | `"append"` | `stage2/text.py:300` |
| `FUSION_PIN_LIBSEQ` | `"1"` | `stage2/pinlib.py:105` |
| `FUSION_PIN_LIBSEQ_FUZZY` | `"0.85"` | `stage2/pinlib.py:108` |
| `FUSION_PIN_LIBSEQ_MARGIN` | `"0.06"` | `stage2/pinlib.py:109` |
| `STAGE1_MERGE_TP` | `"0"` | `stage2/assemble.py:190` |
| `STAGE2_ALT_FILL` | `"0"` | `stage2/assign.py:1702` |
| `STAGE2_BOTH_TOL` | `"0.65"` | `stage2/text.py:87` |
| `STAGE2_BOXIC_LIB` | `""` | `stage2/libpins.py:104` |
| `STAGE2_BOX_ALONE_U1` | `"1"` | `stage2/assemble.py:232` |
| `STAGE2_BOX_GAP_U1` | `"0"` | `stage2/assemble.py:238` |
| `STAGE2_BOX_PIN_DROP_UNREAD` | `"mixed"` | `stage2/assemble.py:264` |
| `STAGE2_BOX_PIN_KEEP_NAMED` | `"0"` | `stage2/assemble.py:301` |
| `STAGE2_BOX_PIN_NEED_NAME` | `"2"` | `stage2/assemble.py:355` |
| `STAGE2_BOX_PIN_PARTIAL` | `"1"` | `stage2/assemble.py:223` |
| `STAGE2_CKPT` | `r"Y:\workspace\EDA\work\ocr_mix\runs\best.pt"` | `stage2/text.py:42` |
| `STAGE2_CKPT_DESIG` | — | `stage2/text.py:505` |
| `STAGE2_CKPT_ROLES` | — | `stage2/text.py:551` |
| `STAGE2_CKPT_ROLES_CKPT` | — | `stage2/text.py:505` |
| `STAGE2_COLOR_NUMBER_CROP` | `"0"` | `stage2/run.py:75` |
| `STAGE2_CONFORM_AUDIT` | `"1"` | `stage2/assign.py:730` |
| `STAGE2_CONFORM_AUDIT_MAX` | `"400"` | `stage2/assign.py:731` |
| `STAGE2_CONFORM_CONF` | `"0.50"` | `stage2/assign.py:718` |
| `STAGE2_CONFORM_GLUE_PEN` | `"0.85"` | `stage2/assign.py:720` |
| `STAGE2_CONFORM_KEY` | `"1"` | `stage2/assign.py:715` |
| `STAGE2_CONFORM_TYPE_PEN` | `"0.85"` | `stage2/assign.py:721` |
| `STAGE2_CPNT_LIB` | `""` | `stage2/libpins.py:103` |
| `STAGE2_DBG_CID` | — | `stage2/assign.py:223` |
| `STAGE2_DBG_PINS` | — *(7 处)* | `stage2/assemble.py:2329` |
| `STAGE2_DEMOTE_CONF` | `"0"` | `stage2/assign.py:1735` |
| `STAGE2_DEMOTE_HARD` | `"0.05"` | `stage2/structmatch.py:36` |
| `STAGE2_DESIG_DBG` | — | `stage2/assign.py:2457` |
| `STAGE2_DESIG_MAXD` | `"0"` | `stage2/assign.py:435` |
| `STAGE2_DESIG_MAXW` | `"6.0"` | `stage2/assign.py:433` |
| `STAGE2_DESIG_RESCUE` | `"0"` | `stage2/assign.py:2397` |
| `STAGE2_DROP_UNKEYED` | `"0"` | `stage2/assemble.py:77` |
| `STAGE2_DROP_UNKEYED_POWER` | `"0"` | `stage2/assemble.py:114` |
| `STAGE2_DUMP_CASE` | `""` *(2 处)* | `stage2/assign.py:1757` |
| `STAGE2_DUMP_KEYPROV` | `"0"` | `stage2/assign.py:1752` |
| `STAGE2_DUMP_PATH` | — *(2 处)* | `stage2/assign.py:1754` |
| `STAGE2_DUMP_READS` | `"0"` *(2 处)* | `stage2/assign.py:744` |
| `STAGE2_EDGE_ANG` | `"6.0"` | `stage2/assemble.py:545` |
| `STAGE2_EDGE_CORNER` | `"12"` | `stage2/assemble.py:958` |
| `STAGE2_EDGE_CORNER_CROSS` | `"1"` | `stage2/assemble.py:960` |
| `STAGE2_EDGE_CORNER_INT` | `"0.05"` | `stage2/assemble.py:959` |
| `STAGE2_EDGE_DEDUP` | `"2.0"` | `stage2/assemble.py:709` |
| `STAGE2_EDGE_DEDUP2` | `"2.0"` | `stage2/assemble.py:714` |
| `STAGE2_EDGE_EXT` | `"20"` | `stage2/assemble.py:752` |
| `STAGE2_EDGE_EXT_ANY` | `"25"` | `stage2/assemble.py:846` |
| `STAGE2_EDGE_EXT_ANY_HIT` | `"2.0"` | `stage2/assemble.py:847` |
| `STAGE2_EDGE_EXT_FREE` | `"0"` | `stage2/assemble.py:785` |
| `STAGE2_EDGE_EXT_FREE_EDGE` | `"3.0"` | `stage2/assemble.py:787` |
| `STAGE2_EDGE_EXT_FREE_PIN` | `"5.0"` | `stage2/assemble.py:786` |
| `STAGE2_EDGE_EXT_HIT` | `"3.0"` | `stage2/assemble.py:753` |
| `STAGE2_EDGE_JOIN` | `"40.0"` | `stage2/assemble.py:544` |
| `STAGE2_EDGE_MERGE` | `"0"` | `stage2/assemble.py:534` |
| `STAGE2_EDGE_OVERLAP` | `"10"` | `stage2/assemble.py:765` |
| `STAGE2_EDGE_OVERLAP_MIN` | `"0.35"` | `stage2/assemble.py:767` |
| `STAGE2_EDGE_OVERLAP_OFF` | `"4.0"` | `stage2/assemble.py:766` |
| `STAGE2_EDGE_SNAP` | `"10.0"` *(2 处)* | `stage2/assemble.py:543` |
| `STAGE2_EDGE_SNAP_ANG` | `"45.0"` | `stage2/assemble.py:614` |
| `STAGE2_EDGE_SNAP_JUNC` | `"0"` | `stage2/assemble.py:615` |
| `STAGE2_EDGE_SPLIT` | `"3.0"` | `stage2/assemble.py:707` |
| `STAGE2_EDGE_SPLIT_COL` | `"0.90"` | `stage2/assemble.py:708` |
| `STAGE2_EDGE_SPLIT_PIN` | `"0"` | `stage2/assemble.py:1373` |
| `STAGE2_EDGE_SPLIT_XNET` | `"0"` | `stage2/assemble.py:1385` |
| `STAGE2_EDGE_SPLIT_XNET_ANG` | `"30"` | `stage2/assemble.py:1387` |
| `STAGE2_EDGE_SPLIT_XNET_MINLEN` | `"6"` | `stage2/assemble.py:1388` |
| `STAGE2_EDGE_TRIM` | `"3.0"` | `stage2/assemble.py:739` |
| `STAGE2_END_TO_BOX` | `"0"` | `stage2/assemble.py:1575` |
| `STAGE2_END_TO_BOX_FREE` | `"3.0"` | `stage2/assemble.py:1578` |
| `STAGE2_END_TO_BOX_MAX` | `"30.0"` | `stage2/assemble.py:1577` |
| `STAGE2_END_TO_BOX_TOL` | `"4.0"` | `stage2/assemble.py:1576` |
| `STAGE2_END_TO_BOX_TYPES` | `"c"` | `stage2/assemble.py:1586` |
| `STAGE2_FAR_CONF` | `"0.50"` | `stage2/assign.py:748` |
| `STAGE2_FAR_DBG` | `"0"` | `stage2/assign.py:1828` |
| `STAGE2_FAR_KEY` | `"0"` | `stage2/assign.py:747` |
| `STAGE2_FAR_REPL` | `"40"` | `stage2/assign.py:757` |
| `STAGE2_FAR_TYPES` | `"box"` | `stage2/assign.py:750` |
| `STAGE2_FREE_STEAL` | `"1"` | `stage2/assign.py:493` |
| `STAGE2_GLOBAL_MATCH` | `"0"` | `stage2/assign.py:1402` |
| `STAGE2_GROW_MIN` | `"0.9"` | `stage2/run.py:39` |
| `STAGE2_GROW_PER_COMP` | `"4"` | `stage2/run.py:41` |
| `STAGE2_KEEP_UNKEYED_IF_NO_DESIG` | `"0"` | `stage2/assemble.py:126` |
| `STAGE2_KEEP_UNKEYED_POWER` | `"0"` | `stage2/assemble.py:120` |
| `STAGE2_KEY_ANYROLE` | `"0"` | `stage2/keyform.py:161` |
| `STAGE2_KEY_CHANNEL` | `"0"` | `stage2/assign.py:1651` |
| `STAGE2_KEY_DBG` | `"0"` | `stage2/assign.py:1655` |
| `STAGE2_KEY_RESCUE` | `"0"` | `stage2/keyform.py:158` |
| `STAGE2_KEY_WCHECK` | `"1"` | `stage2/keyform.py:160` |
| `STAGE2_LED_POLARITY` | `"1"` | `stage2/assemble.py:1829` |
| `STAGE2_LIB_PINS` | `"1"` *(2 处)* | `stage2/assemble.py:399` |
| `STAGE2_LIB_PINS_DBG` | `"0"` | `stage2/assemble.py:401` |
| `STAGE2_LIB_PINS_FUZZY` | `"0.85"` | `stage2/libpins.py:335` |
| `STAGE2_LIB_PINS_NAMEFILL` | `"1"` | `stage2/libpins.py:87` |
| `STAGE2_LIB_PINS_NUMFILL` | `"1"` | `stage2/libpins.py:86` |
| `STAGE2_LIB_PINS_NUMGATE` | `"0"` | `stage2/libpins.py:237` |
| `STAGE2_LIB_REFINE` | `"1"` | `stage2/librefine.py:58` |
| `STAGE2_LIB_REFINE_DEGEN` | `"known2"` *(2 处)* | `stage2/libpins.py:215` |
| `STAGE2_LIB_REFINE_WHEN` | `"any"` | `stage2/librefine.py:64` |
| `STAGE2_MIN_EDGE_LEN` | `"20"` | `stage2/assemble.py:508` |
| `STAGE2_MONO_PIN_OCR` | `PIN_NAME_OCR` | `stage2/run.py:80` |
| `STAGE2_MULTI_H` | `"0.43"` | `stage2/text.py:95` |
| `STAGE2_MULTI_W` | `"0.15"` | `stage2/text.py:94` |
| `STAGE2_NAME_SHAPE` | — | `stage2/assign.py:560` |
| `STAGE2_NEAR_REV` | `""` | `stage2/assign.py:530` |
| `STAGE2_NETLABEL_MIN_H` | `"9"` | `stage2/assign.py:642` |
| `STAGE2_NETLABEL_MIN_W` | `"12"` | `stage2/assign.py:641` |
| `STAGE2_NET_DROP_PWRONLY` | `"0"` | `stage2/assemble.py:471` |
| `STAGE2_NET_DROP_UNKEYED` | `"0"` | `stage2/assemble.py:455` |
| `STAGE2_NET_PAIR` | `"120"` | `stage2/assemble.py:3205` |
| `STAGE2_NET_PAIR_DBG` | — | `stage2/assemble.py:3243` |
| `STAGE2_NET_WELD` | `"16"` | `stage2/assemble.py:590` |
| `STAGE2_NET_WELD_COS` | `"0.5"` | `stage2/assemble.py:592` |
| `STAGE2_NET_WELD_MIN` | `"6.0"` | `stage2/assemble.py:591` |
| `STAGE2_NET_WELD_TPAD` | `"3.0"` | `stage2/assemble.py:593` |
| `STAGE2_NODELETE` | `"1"` | `stage2/structmatch.py:34` |
| `STAGE2_NO_BORROW` | — | `stage2/assign.py:1140` |
| `STAGE2_OWN_ASSIGN` | `"0"` | `stage2/assign.py:208` |
| `STAGE2_OWN_BBOX` | `"1"` | `stage2/assign.py:189` |
| `STAGE2_OWN_MATCH` | `"0"` | `stage2/assign.py:219` |
| `STAGE2_PAD_FRAC` | `"0.0"` | `stage2/text.py:70` |
| `STAGE2_PAD_FRAC_Y` | `"0.20"` | `stage2/text.py:79` |
| `STAGE2_PAD_MAX` | `"99.0"` | `stage2/text.py:71` |
| `STAGE2_PAD_MIN` | `"2.0"` | `stage2/text.py:69` |
| `STAGE2_PAD_MIN_Y` | `"3.0"` | `stage2/text.py:80` |
| `STAGE2_PARTNAME_FREE_H` | `"18.0"` | `stage2/assign.py:380` |
| `STAGE2_PART_STEAL` | `"0"` | `stage2/assign.py:510` |
| `STAGE2_PART_STEAL_CONF` | `"0.55"` | `stage2/assign.py:511` |
| `STAGE2_PART_STEAL_PEN` | `"0.35"` | `stage2/assign.py:512` |
| `STAGE2_PHANTOM_DBG` | — | `stage2/assemble.py:3201` |
| `STAGE2_PHANTOM_MERGE` | — | `stage2/assemble.py:3156` |
| `STAGE2_PHANTOM_NET` | `"0"` | `stage2/assemble.py:3133` |
| `STAGE2_PINCOL` | `"0"` | `stage2/assign.py:480` |
| `STAGE2_PINCOL_COVER` | `"0.5"` | `stage2/assign.py:481` |
| `STAGE2_PINCOL_WIN` | `"1"` | `stage2/run.py:45` |
| `STAGE2_PINCOL_WIN_KEYS` | `"out_win"` | `stage2/run.py:48` |
| `STAGE2_PINNAME_CFRAC` | `"0.20"` | `stage2/assign.py:358` |
| `STAGE2_PINNAME_DUPGAP` | `"6.0"` | `stage2/assign.py:399` |
| `STAGE2_PINNAME_DUPOV` | `"0.60"` | `stage2/assign.py:398` |
| `STAGE2_PINNAME_KEYLIKE` | `"0.2"` | `stage2/assign.py:366` |
| `STAGE2_PINNAME_MAXW` | `"6.0"` | `stage2/assign.py:348` |
| `STAGE2_PINNAME_NODIGIT` | `"0"` | `stage2/assign.py:300` |
| `STAGE2_PINNAME_OUTSIDE` | `"0"` | `stage2/assign.py:423` |
| `STAGE2_PINNAME_RESCUE` | `"1"` | `stage2/pinname.py:67` |
| `STAGE2_PINNAME_RESCUE_AGREE` | `"1"` | `stage2/pinname.py:71` |
| `STAGE2_PINNAME_RESCUE_CONF` | `"0.70"` | `stage2/pinname.py:69` |
| `STAGE2_PINNAME_RESCUE_DEPTH` | `"52.0"` | `stage2/pinname.py:74` |
| `STAGE2_PINNAME_RESCUE_GRAY` | `"170"` | `stage2/pinname.py:76` |
| `STAGE2_PINNAME_RESCUE_INSET` | `"4.0"` | `stage2/pinname.py:73` |
| `STAGE2_PINNAME_RESCUE_SAT` | `"70"` | `stage2/pinname.py:77` |
| `STAGE2_PINNAME_ROWNEAR` | `"1"` | `stage2/assign.py:396` |
| `STAGE2_PINNAME_ROWTIE` | `"1.5"` | `stage2/assign.py:397` |
| `STAGE2_PINNUM_CONSENSUS` | `"0"` | `stage2/assign.py:61` |
| `STAGE2_PINNUM_DECAY` | `"10"` | `stage2/assign.py:253` |
| `STAGE2_PINNUM_DISTW` | `"0.005"` | `stage2/assign.py:240` |
| `STAGE2_PINNUM_DUP` | `"0"` | `stage2/assign.py:265` |
| `STAGE2_PINNUM_INSIDE` | `"0"` | `stage2/assign.py:299` |
| `STAGE2_PINNUM_NEAR` | `"60"` | `stage2/assign.py:236` |
| `STAGE2_PINNUM_PAD_FRAC_Y` | `"0.0"` | `stage2/text.py:125` |
| `STAGE2_PINNUM_PXC` | `"0"` | `stage2/assign.py:373` |
| `STAGE2_PINNUM_STEAL` | `"0"` | `stage2/assign.py:280` |
| `STAGE2_PINNUM_TIGHT` | `"1"` | `stage2/text.py:122` |
| `STAGE2_PINNUM_TIGHT_DARK` | `"160"` | `stage2/text.py:124` |
| `STAGE2_PINNUM_TIGHT_SAT` | `"60"` | `stage2/text.py:123` |
| `STAGE2_PINROT` | `"0"` | `stage2/pinrot.py:73` |
| `STAGE2_PINROT_CONF` | `"0.70"` | `stage2/pinrot.py:75` |
| `STAGE2_PINROT_HALF` | `"14.0"` | `stage2/pinrot.py:80` |
| `STAGE2_PINROT_MIN_NAME` | `"2"` | `stage2/pinrot.py:93` |
| `STAGE2_PINROT_NAME` | `"1"` | `stage2/pinrot.py:86` |
| `STAGE2_PINROT_NAME_DEPTH` | `"46.0"` | `stage2/pinrot.py:77` |
| `STAGE2_PINROT_NUM` | `"1"` | `stage2/pinrot.py:87` |
| `STAGE2_PINROT_NUM_DEPTH` | `"34.0"` | `stage2/pinrot.py:78` |
| `STAGE2_PINROT_OVERWRITE` | `"1"` | `stage2/pinrot.py:84` |
| `STAGE2_PINROT_RAW` | `"0"` | `stage2/pinrot.py:89` |
| `STAGE2_PINROT_SHAPE_ROUND` | `"8.0"` | `stage2/pinrot.py:91` |
| `STAGE2_PINROT_WIN_PAD` | `"60.0"` | `stage2/pinrot.py:82` |
| `STAGE2_PIN_ALONG` | `"1"` | `stage2/assign.py:445` |
| `STAGE2_PIN_ANCHOR` | `"12"` | `stage2/structmatch.py:54` |
| `STAGE2_PIN_AXIS_DL` | `"0"` | `stage2/assemble.py:1813` |
| `STAGE2_PIN_BODY` | `"0"` | `stage2/assemble.py:2939` |
| `STAGE2_PIN_BODY_DBG` | — | `stage2/assemble.py:2986` |
| `STAGE2_PIN_BODY_GRID` | `"1"` | `stage2/assemble.py:2941` |
| `STAGE2_PIN_COVER` | `"0.15"` | `stage2/assign.py:449` |
| `STAGE2_PIN_DROP_LAUNDERED` | `"1"` | `stage2/assemble.py:141` |
| `STAGE2_PIN_DROP_UNKEYED` | `"0"` | `stage2/assemble.py:105` |
| `STAGE2_PIN_GREY_EVIDENCE` | `"1"` | `stage2/run.py:96` |
| `STAGE2_PIN_LANE_CLEANUP` | `"light"` | `stage2/run.py:78` |
| `STAGE2_PIN_LANE_CROP` | `"1"` | `stage2/run.py:77` |
| `STAGE2_PIN_LEAD_ATTACH` | `"1"` | `stage2/assemble.py:3356` |
| `STAGE2_PIN_LEAD_DEG` | `"35.0"` | `stage2/assemble.py:3360` |
| `STAGE2_PIN_LEAD_MOVE` | `"1"` | `stage2/assemble.py:3357` |
| `STAGE2_PIN_LEAD_MOVE_R` | `"50.0"` | `stage2/assemble.py:3359` |
| `STAGE2_PIN_LEAD_R` | `"40.0"` | `stage2/assemble.py:3358` |
| `STAGE2_PIN_NAME_OCR` | `"off"` | `stage2/run.py:79` |
| `STAGE2_PIN_NUMBER_REPAIR` | `"0"` | `stage2/run.py:81` |
| `STAGE2_PIN_OFFSET` | `"0"` | `stage2/assemble.py:2878` |
| `STAGE2_PIN_OFFSET_DBG` | — *(2 处)* | `stage2/assemble.py:2928` |
| `STAGE2_PIN_OFFSET_MIN` | `"2"` | `stage2/assemble.py:2879` |
| `STAGE2_PIN_RESIDUAL` | `"0"` | `stage2/assign.py:457` |
| `STAGE2_PIN_RESIDUAL_MIN` | `"0"` | `stage2/assign.py:458` |
| `STAGE2_PIN_SNAP` | `"0"` *(2 处)* | `stage2/assemble.py:861` |
| `STAGE2_PIN_SNAP_DBG` | — *(2 处)* | `stage2/assemble.py:2871` |
| `STAGE2_PIN_SNAP_N` | `"0"` | `stage2/assemble.py:2993` |
| `STAGE2_PIN_SNAP_PERP` | `"8"` | `stage2/assemble.py:863` |
| `STAGE2_PIN_SNAP_R` | `"45"` | `stage2/assemble.py:862` |
| `STAGE2_PIN_STAMP` | `"1"` | `stage2/assemble.py:1811` |
| `STAGE2_PIN_STUB_MERGE` | `"12"` | `stage2/assign.py:344` |
| `STAGE2_PIN_TEXT_CROP` | `"0"` | `stage2/run.py:76` |
| `STAGE2_PIN_TRUST_DROP` | `""` | `stage2/run.py:103` |
| `STAGE2_PREFIX` | `"strict"` | `stage2/assign.py:159` |
| `STAGE2_PREFIX_SKEL` | `"0"` | `stage2/assign.py:169` |
| `STAGE2_RAIL_ATTACH` | `"1"` | `stage2/assemble.py:3319` |
| `STAGE2_RAIL_SPLIT` | `"1"` | `stage2/assemble.py:3371` |
| `STAGE2_SHAPE_FILL` | `"0"` | `stage2/assign.py:1497` |
| `STAGE2_SHAPE_MINCONF` | `"0.90"` | `stage2/assign.py:1499` |
| `STAGE2_SHAPE_PEN` | `"0.60"` | `stage2/assign.py:1498` |
| `STAGE2_SLOT_CLEAN` | `"1"` | `stage2/assign.py:2500` |
| `STAGE2_STAMP_ONE_SIDE` | `"1"` | `stage2/assemble.py:1871` |
| `STAGE2_STEAL` | `"80"` | `stage2/assign.py:491` |
| `STAGE2_SYM_MERGE` | `"0"` | `stage2/assemble.py:494` |
| `STAGE2_SYM_VERTEX` | `"40"` | `stage2/assemble.py:2783` |
| `STAGE2_SYM_VERTEX_DBG` | — *(2 处)* | `stage2/assemble.py:2826` |
| `STAGE2_TEXT_INK` | `"0"` | `stage2/text.py:141` |
| `STAGE2_TEXT_INK_DARK` | `"160"` | `stage2/text.py:143` |
| `STAGE2_TEXT_INK_SAT` | `"60"` | `stage2/text.py:142` |
| `STAGE2_TIGHT_ROLES` | `"pin_number,pin_name"` | `stage2/text.py:129` |
| `STAGE2_TP_CV_SUBTYPES` | `"ring3"` | `stage2/assemble.py:192` |
| `STAGE2_TP_LEDGER` | `"0"` | `stage2/assemble.py:207` |
| `STAGE2_TP_TRUST_EVIDENCE` | `"1"` | `stage2/assemble.py:204` |
| `STAGE2_TRACE_DUMP` | `""` | `stage2/assemble.py:3276` |
| `STAGE2_UNKEYED_GATE` | `"1"` | `stage2/assemble.py:2269` |
| `STAGE2_UNKEYED_GATE_N` | `"2"` | `stage2/assemble.py:2270` |
| `STAGE2_VALUE_DISTW` | `"0.035"` | `stage2/assign.py:623` |
| `STAGE2_VALUE_FAMILY` | `"0"` | `stage2/assign.py:1013` |
| `STAGE2_VALUE_FILL` | `"0"` | `stage2/assign.py:1071` |
| `STAGE2_VALUE_FILL_GAIN` | `"0"` | `stage2/assign.py:1072` |
| `STAGE2_VALUE_NETNAME` | `"0"` | `stage2/assign.py:612` |
| `STAGE2_VALUE_STEAL` | `"40"` | `stage2/assign.py:502` |
| `STAGE2_VALUE_TRIM` | `"0"` | `stage2/assign.py:599` |
| `STAGE2_VALUE_UNIT` | `"1"` | `stage2/assign.py:587` |
| `STAGE2_VALUE_UNITX` | `"0"` | `stage2/assign.py:1014` |
| `STAGE2_WIDE_TOL` | `"0.5"` | `stage2/text.py:302` |
| `TEXT_LINESPLIT` | `"1"` | `stage2/text.py:145` |
| `TEXT_LINESPLIT_H` | `"1.7"` | `stage2/text.py:146` |

<a id="stage3"></a>

## stage3

| 开关 | 默认值 | 定义位置 |
|---|---|---|
| `STAGE3_AUTO_LLM` | `"0"` | `stage3/arbitrate.py:93` |
| `STAGE3_OWN` | `"on"` | `stage3/arbitrate.py:172` |

<a id="fusion"></a>

## fusion

| 开关 | 默认值 | 定义位置 |
|---|---|---|
| `AXIS_CHAR_CAL` | — | `fusion/axis.py:133` |
| `AXIS_CHAR_SIG` | — | `fusion/axis.py:130` |
| `FUSION_ASSIGNED_DIR` | `str(RUN_DIR / "stage2" / "_assigned"` | `fusion/run.py:298` |
| `FUSION_AUDIT` | `"1"` | `fusion/audit.py:76` |
| `FUSION_AUDIT_FULLW` | `"880"` | `fusion/audit.py:77` |
| `FUSION_AUDIT_KINDS` | `"net_short,net_rail_merge,key_duplicate,key_type_mismatch,va` | `fusion/audit.py:82` |
| `FUSION_AUDIT_MAX` | `"16"` | `fusion/audit.py:69` |
| `FUSION_AUDIT_MAXH` | `"1400"` | `fusion/audit.py:78` |
| `FUSION_AUDIT_REREAD` | `"1"` | `fusion/audit.py:74` |
| `FUSION_AUDIT_SCORE` | `"1.30"` | `fusion/audit.py:68` |
| `FUSION_AXIS` | `"1"` | `fusion/axis.py:87` |
| `FUSION_AXIS_VERT` | `"0"` | `fusion/axis.py:155` |
| `FUSION_BASE` | `"ours"` | `fusion/run.py:239` |
| `FUSION_BOXSCAN` | `"0"` | `fusion/claims.py:151` |
| `FUSION_BOX_PIN_PRIOR` | `"0"` | `fusion/priors.py:135` |
| `FUSION_CASES` | `""` | `fusion/run.py:580` |
| `FUSION_COMPDEDUP` | `"1"` | `fusion/dedup.py:56` |
| `FUSION_CONV_EXACT` | `"1"` | `fusion/run.py:98` |
| `FUSION_CROPFIX` | `"1"` | `fusion/cropfix.py:61` |
| `FUSION_CROPFIX_LADDER` | `"0.5,1.0,2.0,4.0"` | `fusion/cropfix.py:54` |
| `FUSION_CROPFIX_MAX` | `"24"` | `fusion/cropfix.py:57` |
| `FUSION_CROPFIX_ROUNDS` | `"2"` | `fusion/cropfix.py:56` |
| `FUSION_CROPFIX_SCALE` | `os.environ.get("VLM_SCALE", "4"` | `fusion/cropfix.py:50` |
| `FUSION_CROPFIX_SCORE` | `"1.25"` | `fusion/cropfix.py:58` |
| `FUSION_EDGECAL` | `"1"` | `fusion/edgecal.py:61` |
| `FUSION_EDGECAL_NEAR` | `3.5` | `fusion/edgecal.py:64` |
| `FUSION_EDGECAL_TRIM` | `1.0` | `fusion/edgecal.py:62` |
| `FUSION_EDGECAL_TRIM_NEAR` | `4.0` | `fusion/edgecal.py:63` |
| `FUSION_FREE_SCORE` | `"0.5"` | `fusion/llm_judge.py:61` |
| `FUSION_GAP_COVER` | `"0.45"` | `fusion/solve.py:64` |
| `FUSION_GAP_FILL` | `"1"` | `fusion/solve.py:69` |
| `FUSION_GAP_MIN_RUN` | `"4"` | `fusion/solve.py:62` |
| `FUSION_GAP_SCORE` | `"0.90"` | `fusion/solve.py:66` |
| `FUSION_ITERATE` | `"1"` | `fusion/iterate.py:58` |
| `FUSION_ITERATE_COLS` | `"4"` | `fusion/iterate.py:63` |
| `FUSION_ITERATE_KEEP_OURS` | `"0"` | `fusion/iterate.py:70` |
| `FUSION_ITERATE_MAX_PINS` | `"120"` | `fusion/iterate.py:67` |
| `FUSION_ITERATE_MIN_NAME_LEN` | `"2"` | `fusion/iterate.py:72` |
| `FUSION_ITERATE_MIN_PINS` | `"4"` | `fusion/iterate.py:66` |
| `FUSION_ITERATE_NUM_WIN` | `"out_win"` | `fusion/iterate.py:61` |
| `FUSION_ITERATE_PAD` | `"3"` | `fusion/iterate.py:65` |
| `FUSION_ITERATE_PER` | `"8"` | `fusion/iterate.py:62` |
| `FUSION_ITERATE_SCORE` | `"0.9"` | `fusion/iterate.py:75` |
| `FUSION_ITERATE_SIDE` | `"150"` | `fusion/iterate.py:64` |
| `FUSION_ITERATE_WIN` | `"in_win"` | `fusion/iterate.py:60` |
| `FUSION_ITER_GAP` | `"2"` | `fusion/iterate.py:288` |
| `FUSION_JUDGE` | `"1"` | `fusion/llm_judge.py:71` |
| `FUSION_JUDGE_CELL` | `"260"` | `fusion/llm_judge.py:739` |
| `FUSION_JUDGE_COLS` | `"3"` | `fusion/llm_judge.py:740` |
| `FUSION_JUDGE_CROP` | `"tight"` | `fusion/llm_judge.py:73` |
| `FUSION_JUDGE_CTX` | `"1"` | `fusion/llm_judge.py:511` |
| `FUSION_JUDGE_CTX_MAXCH` | `"2400"` | `fusion/llm_judge.py:518` |
| `FUSION_JUDGE_CTX_NB` | `"6"` | `fusion/llm_judge.py:516` |
| `FUSION_JUDGE_CTX_SEV` | `"0.5"` | `fusion/llm_judge.py:523` |
| `FUSION_JUDGE_CTX_SUS` | `"2"` | `fusion/llm_judge.py:524` |
| `FUSION_JUDGE_MAX` | `"64"` | `fusion/llm_judge.py:65` |
| `FUSION_JUDGE_MISMATCH` | `"0.30"` | `fusion/llm_judge.py:75` |
| `FUSION_JUDGE_MODE` | `"auto"` | `fusion/llm_judge.py:135` |
| `FUSION_JUDGE_OTHER` | `"1"` | `fusion/llm_judge.py:136` |
| `FUSION_JUDGE_SCALE` | `os.environ.get("VLM_SCALE", "4"` | `fusion/llm_judge.py:63` |
| `FUSION_JUDGE_SHEET` | `"1"` | `fusion/llm_judge.py:738` |
| `FUSION_JUDGE_SHEET_MAX` | `"36"` | `fusion/llm_judge.py:741` |
| `FUSION_JUDGE_SHEET_MIN` | `"9999"` | `fusion/llm_judge.py:745` |
| `FUSION_JUDGE_WHY` | `"1"` | `fusion/llm_judge.py:514` |
| `FUSION_LLM` | `"1"` | `fusion/run.py:94` |
| `FUSION_LLM_SCORE` | `"1.5"` | `fusion/llm_judge.py:45` |
| `FUSION_MAX_CANDS` | `"8"` | `fusion/solve.py:73` |
| `FUSION_MAX_OPEN` | `"24"` | `fusion/solve.py:49` |
| `FUSION_MERGE_PIN_MIN` | `"8"` | `fusion/run.py:268` |
| `FUSION_MIN_NET_PINS` | `"2"` | `fusion/plausible.py:57` |
| `FUSION_MODEL_SET` | `"1"` | `fusion/modelset.py:85` |
| `FUSION_MODEL_SET_MARGIN` | `"2"` | `fusion/modelset.py:87` |
| `FUSION_MODEL_SET_MIN` | `"2"` | `fusion/modelset.py:86` |
| `FUSION_NAME_SNAP` | `"1"` | `fusion/namefix.py:116` |
| `FUSION_NAME_SNAP_MINLEN` | `"4"` | `fusion/namefix.py:118` |
| `FUSION_NAME_SNAP_TOL` | `"4"` | `fusion/namefix.py:117` |
| `FUSION_OPEN_FREEFORM` | `"1"` | `fusion/solve.py:71` |
| `FUSION_OPEN_FREEFORM_MAX` | `"8"` | `fusion/solve.py:60` |
| `FUSION_OPEN_MARGIN` | `"0.10"` | `fusion/solve.py:47` |
| `FUSION_OUT_DIR` | `str(ROOT / "work" / "_fusion_out"` | `fusion/run.py:75` |
| `FUSION_PIN_LIBFIX` | `"1"` | `fusion/pinfix.py:74` |
| `FUSION_PIN_LIBFIX_MARGIN` | `"0.05"` | `fusion/pinfix.py:190` |
| `FUSION_PIN_LIBFIX_RATIO` | `"0.75"` | `fusion/pinfix.py:186` |
| `FUSION_PWRCLIP` | `"1"` | `fusion/pwrclip.py:74` |
| `FUSION_READ_DROP1` | `"0.40"` | `fusion/priors.py:426` |
| `FUSION_READ_DROP1O` | `"0.12"` | `fusion/priors.py:427` |
| `FUSION_READ_DROPM` | `"0.12"` | `fusion/priors.py:428` |
| `FUSION_READ_EXACT` | `"1.00"` | `fusion/priors.py:425` |
| `FUSION_READ_FAR` | `"0.01"` | `fusion/priors.py:431` |
| `FUSION_READ_INS` | `"0.30"` | `fusion/priors.py:429` |
| `FUSION_READ_ROW_OVERLAP` | `"0.5"` | `fusion/priors.py:282` |
| `FUSION_READ_SUB` | `"0.05"` | `fusion/priors.py:430` |
| `FUSION_REF_PRUNE` | `"1"` | `fusion/run.py:117` |
| `FUSION_REF_PRUNE_DESIG` | `"1"` | `fusion/run.py:118` |
| `FUSION_REGION_MIN` | `"120"` | `fusion/claims.py:114` |
| `FUSION_REGION_PAD` | `"24"` | `fusion/claims.py:112` |
| `FUSION_REGION_TOPK` | `"1"` | `fusion/claims.py:156` |
| `FUSION_REPAIR_PENALTY` | `"0.85"` | `fusion/priors.py:223` |
| `FUSION_RESCAN_CROP` | `"window"` | `fusion/rescan.py:42` |
| `FUSION_RESCAN_MAX` | `"12"` | `fusion/rescan.py:40` |
| `FUSION_RESCAN_SCORE` | `"1.2"` | `fusion/rescan.py:38` |
| `FUSION_RESCAN_VALUE` | `"0"` | `fusion/rescan.py:98` |
| `FUSION_ROUNDS` | `"2"` | `fusion/run.py:82` |
| `FUSION_RUN_ALTWINDOW` | `"0"` | `fusion/priors.py:247` |
| `FUSION_RUN_DIR` | `r"Y:\workspace\EDA\work\_submit_out220r36\runs"` *(3 处)* | `fusion/audit.py:554` |
| `FUSION_RUN_ORDER` | `"1"` | `fusion/priors.py:259` |
| `FUSION_RUN_ORDER_AGREE` | `"0.6"` | `fusion/priors.py:278` |
| `FUSION_RUN_ORDER_BONUS` | `"0.05"` | `fusion/priors.py:275` |
| `FUSION_RUN_ORDER_HOME` | `"0"` | `fusion/priors.py:270` |
| `FUSION_RUN_ORDER_MIN_TRUST` | `"2"` | `fusion/priors.py:280` |
| `FUSION_SEV_KEY` | `"0.70"` | `fusion/plausible.py:49` |
| `FUSION_SEV_RAIL` | `"0.60"` | `fusion/plausible.py:47` |
| `FUSION_SEV_SHORT` | `"0.90"` | `fusion/plausible.py:45` |
| `FUSION_SEV_VAL` | `"0.25"` | `fusion/plausible.py:51` |
| `FUSION_SHAPE_FLOOR` | `"0.35"` | `fusion/priors.py:165` |
| `FUSION_SHAPE_GATE` | `"1"` | `fusion/priors.py:122` |
| `FUSION_SHAPE_SCORE` | `""` | `fusion/priors.py:124` |
| `FUSION_SHAPE_W` | `"0.0"` | `fusion/priors.py:164` |
| `FUSION_SHEET_OUT` | `r"Y:\workspace\EDA\work\preview\_judge_sheet.png"` | `fusion/llm_judge.py:856` |
| `FUSION_SHEET_TIMEOUT` | `"300"` | `fusion/llm_judge.py:878` |
| `FUSION_STAGE1_DIR` | `str(RUN_DIR / "stage1"` *(2 处)* | `fusion/run.py:296` |
| `FUSION_TOP_WINDOWS` | `"3"` | `fusion/priors.py:225` |
| `FUSION_VALUE_FIX` | `"1"` | `fusion/valfix.py:66` |
| `FUSION_VALUE_MIN_N` | `"4"` | `fusion/plausible.py:55` |
| `FUSION_VALUE_RATIO` | `"4.0"` | `fusion/plausible.py:53` |
| `FUSION_VLM_DIR` | `r"Y:\workspace\EDA\work\_fusion_vlm"` *(2 处)* | `fusion/audit.py:556` |
| `FUSION_VLM_PIN_ADDONLY` | `"1"` | `fusion/claims.py:105` |
| `FUSION_VLM_PIN_NAME` | `"1"` | `fusion/claims.py:83` |
| `FUSION_VLM_PIN_NUM` | `"1"` | `fusion/claims.py:82` |
| `FUSION_VLM_PIN_SCORE` | `"1.15"` | `fusion/claims.py:74` |
| `FUSION_VLM_SCORE` | `"1.15"` | `fusion/claims.py:71` |
| `FUSION_WORKERS` | `"1"` | `fusion/run.py:602` |
| `VLM_TIMEOUT` | `"600"` *(3 处)* | `fusion/audit.py:75` |

<a id="vlm"></a>

## vlm

| 开关 | 默认值 | 定义位置 |
|---|---|---|
| `FUSION_PINNAME` | `"1"` | `vlm/pinname.py:100` |
| `FUSION_PINNAME_CALLS` | `120` *(2 处)* | `vlm/pinname.py:109` |
| `FUSION_PINNAME_CONF` | `0.85` | `vlm/pinname.py:101` |
| `FUSION_PINNAME_GUARD` | `"1"` | `vlm/pinname.py:110` |
| `FUSION_PINNAME_LONGCONF` | `0.95` | `vlm/pinname.py:105` |
| `FUSION_PINNAME_LONGW` | `40.0` | `vlm/pinname.py:104` |
| `FUSION_PINNAME_MAX` | `60` *(2 处)* | `vlm/pinname.py:108` |
| `FUSION_PINNAME_MAXW` | `10.0` | `vlm/pinname.py:102` |
| `FUSION_PINNAME_PAD` | `PAD` | `vlm/pinname.py:247` |
| `FUSION_PINNAME_SCALE` | `4` *(2 处)* | `vlm/pinname.py:107` |
| `FUSION_PINZOOM` | `"0"` | `vlm/pinzoom.py:85` |
| `FUSION_PINZOOM_BATCH` | `BATCH` | `vlm/pinzoom.py:288` |
| `FUSION_PINZOOM_CALLS` | `MAX_CALLS` | `vlm/pinzoom.py:289` |
| `FUSION_PINZOOM_IC_ONLY` | `"1"` | `vlm/pinzoom.py:290` |
| `FUSION_PINZOOM_MAX` | `MAX_SLOTS` | `vlm/pinzoom.py:287` |
| `FUSION_PINZOOM_RETRY` | `2` | `vlm/pinzoom.py:291` |
| `FUSION_VALREAD` | `"1"` | `vlm/valread.py:84` |
| `FUSION_VALREAD_CALLS` | `80` *(2 处)* | `vlm/valread.py:90` |
| `FUSION_VALREAD_CONF` | — | `vlm/valread.py:87` |
| `FUSION_VALREAD_GUARD` | `"1"` | `vlm/valread.py:91` |
| `FUSION_VALREAD_MAX` | `40` *(2 处)* | `vlm/valread.py:89` |
| `FUSION_VALREAD_PAD` | `PAD` | `vlm/valread.py:201` |
| `FUSION_VALREAD_SCALE` | `4` *(2 处)* | `vlm/valread.py:88` |
| `FUSION_VALREAD_TRIGGER` | — | `vlm/valread.py:85` |
| `FUSION_VALREAD_UF_ONLY` | `"0"` | `vlm/valread.py:86` |
| `LLM_API_KEY` | `""` | `vlm/client.py:76` |
| `LLM_API_KEY_FILE` | `""` | `vlm/client.py:80` |
| `LLM_BASE_URL` | `"http://127.0.0.1:1234/v1"` | `vlm/client.py:59` |
| `LLM_MODEL` | `"qwen3.5-9b"` | `vlm/client.py:106` |
| `VLM_BUDGET_S` | `"0"` | `vlm/client.py:113` |
| `VLM_DESIG_DBG` | — *(3 处)* | `vlm/enrich.py:708` |
| `VLM_DESIG_GUARD` | `"1"` | `vlm/enrich.py:317` |
| `VLM_DESIG_PICKLOG` | — | `vlm/enrich.py:714` |
| `VLM_DESIG_READLOG` | — | `vlm/enrich.py:650` |
| `VLM_DESIG_RETRY` | `"0"` | `vlm/enrich.py:306` |
| `VLM_DESIG_RETRY_MAX` | `"12"` | `vlm/enrich.py:308` |
| `VLM_DESIG_RETRY_SCALE` | `"3.0"` | `vlm/enrich.py:309` |
| `VLM_DESIG_SCALE` | `"2.0"` | `vlm/enrich.py:287` |
| `VLM_MAX_TOKENS` | `"2048"` | `vlm/client.py:108` |
| `VLM_PIN_CHANNEL` | `"0"` | `vlm/enrich.py:1023` |
| `VLM_PIN_COLS` | `"4"` | `vlm/enrich.py:1032` |
| `VLM_PIN_MERGE` | `"0"` | `vlm/enrich.py:987` |
| `VLM_PIN_NUM_RE` | `r"^\d{1,3}$"` | `vlm/enrich.py:990` |
| `VLM_PIN_PER` | `"8"` | `vlm/enrich.py:1031` |
| `VLM_PIN_REBUILD` | `"1"` | `vlm/enrich.py:997` |
| `VLM_PIN_SCALE` | `"1.0"` | `vlm/enrich.py:590` |
| `VLM_REASONING` | `"none"` | `vlm/client.py:109` |
| `VLM_RETRY` | `"3"` | `vlm/client.py:122` |
| `VLM_RETRY_BASE` | `"1.0"` | `vlm/client.py:123` |
| `VLM_SCALE` | `"4"` | `vlm/client.py:107` |
| `VLM_TIMEOUT` | `"600"` | `vlm/client.py:110` |

<a id="textrec"></a>

## textrec

| 开关 | 默认值 | 定义位置 |
|---|---|---|
| `TEXTREC_BUCKET` | `"0"` | `textrec/infer.py:49` |
| `TEXTREC_BUCKET_MAX` | `"64"` | `textrec/infer.py:113` |
| `TEXTREC_CUDNN_RNN` | — | `textrec/model.py:28` |
| `TEXTREC_EXTRA_STEPS` | `"0"` | `textrec/infer.py:53` |
| `TEXTREC_INK_EMPTY_RUN` | `"2"` | `textrec/config.py:100` |
| `TEXTREC_INK_FRAME` | `"2.5"` | `textrec/config.py:108` |
| `TEXTREC_INK_LINE_FRAC` | `"0.85"` | `textrec/config.py:102` |
| `TEXTREC_INK_MIN_FRAC` | `"0.10"` | `textrec/config.py:99` |
| `TEXTREC_INK_PROBE_PAD` | `"4"` | `textrec/config.py:86` |
| `TEXTREC_INK_PROBE_PAD_FRAC` | `"0.25"` | `textrec/config.py:104` |
| `TEXTREC_INK_RUN_FRAC` | `"0.5"` | `textrec/config.py:98` |
| `TEXTREC_INK_THRESH` | `"160"` | `textrec/config.py:84` |
| `TEXTREC_INK_YNORM` | `"0"` | `textrec/config.py:82` |

<a id="cvparts"></a>

## cvparts

| 开关 | 默认值 | 定义位置 |
|---|---|---|
| `CV_AGENDA_MAX` | `"400"` | `cvparts/unified.py:49` |
| `CV_DESIG_NEAR` | `"60"` | `cvparts/unified.py:51` |
| `CV_GRID_CELL` | `"32"` | `cvparts/unified.py:48` |
| `CV_TP_DOT` | `"1"` | `cvparts/testpoint.py:91` |
| `CV_UNIFIED` | `"0"` | `cvparts/unified.py:47` |
| `EDA_TEXTBOX_INK` | `170` | `cvparts/textbox.py:63` |
| `OWN_FLOOR_PX` | `"0.0"` | `cvparts/own.py:79` |
| `OWN_FRAC` | `"0.30"` | `cvparts/own.py:78` |
| `OWN_LEGACY_PX` | `"12.0"` | `cvparts/own.py:81` |
| `TEXTSEG_GAP` | `"1.2"` | `cvparts/textseg.py:71` |
| `TEXTSEG_H_MULT` | `"2.2"` | `cvparts/textseg.py:82` |

<a id="pipeline"></a>

## pipeline

| 开关 | 默认值 | 定义位置 |
|---|---|---|
| `EDA_PCB_REPO` | — | `pipeline/config.py:23` |
| `EDA_ROOT` | — | `pipeline/config.py:25` |
| `PSKEL_NODE_SOLVE` | `"1"` | `pipeline/pskel.py:996` |
| `PSKEL_NODE_SOLVE_FIT` | `"4.0"` | `pipeline/pskel.py:1004` |
| `PSKEL_NODE_SOLVE_LAM` | `"0.05"` | `pipeline/pskel.py:998` |
| `STAGE1_WIRE_CLOSE_K` | `"3" if self.close_thin else "0"` | `pipeline/pskel.py:1160` |
| `STAGE1_WIRE_COLOUR` | `"confirm"` | `pipeline/pskel.py:358` |
| `STAGE1_WIRE_CROSS_FILL` | `"1"` | `pipeline/pskel.py:966` |
| `STAGE1_WIRE_CROSS_FILL_AXES` | `"2"` | `pipeline/pskel.py:972` |
| `STAGE1_WIRE_CROSS_FILL_R` | `"8.0"` | `pipeline/pskel.py:968` |
| `STAGE1_WIRE_DEBUG` | — *(2 处)* | `pipeline/pskel.py:381` |
| `STAGE1_WIRE_EDGE` | `"0"` | `pipeline/pskel.py:929` |
| `STAGE1_WIRE_EDGE_MIN` | `"10.0"` | `pipeline/pskel.py:1127` |
| `STAGE1_WIRE_EDGE_NEAR` | `"2"` | `pipeline/pskel.py:1140` |
| `STAGE1_WIRE_EDGE_REL` | `"0.55"` | `pipeline/pskel.py:1123` |
| `STAGE1_WIRE_GAPBRIDGE` | `"0"` | `pipeline/pskel.py:949` |
| `STAGE1_WIRE_GAPBRIDGE_ANG` | `"35.0"` | `pipeline/pskel.py:955` |
| `STAGE1_WIRE_GAPBRIDGE_CROSS` | `"1"` | `pipeline/pskel.py:958` |
| `STAGE1_WIRE_GAPBRIDGE_MAX` | `"12.0"` | `pipeline/pskel.py:951` |
| `STAGE1_WIRE_UNCONF` | `"veto"` | `pipeline/pskel.py:1093` |
| `STAGE1_WIRE_UNCONF_NEAR` | `"3"` | `pipeline/pskel.py:1107` |
| `WINDIR` | `r'C:\Windows'` | `pipeline/pglyph.py:30` |

<a id="submission"></a>

## submission

| 开关 | 默认值 | 定义位置 |
|---|---|---|
| `EDA_CASE_WORKERS` | `""` | `submission/run.py:448` |
| `EDA_PUBLISH_S` | `"15"` | `submission/run.py:754` |
| `EDA_TIER_ARGS` | `"-lean"` | `submission/agent_system:53` |
| `EDA_WALL_BUDGET_S` | `"0"` | `submission/run.py:747` |
| `FUSION_DESEG` | `"1"` | `submission/run.py:1514` |
| `FUSION_DESEG_DUP` | `"1"` | `submission/run.py:1516` |
| `FUSION_DESEG_MINLEN` | `"6.0"` | `submission/run.py:1521` |
| `FUSION_DESEG_SPLIT` | `"1"` | `submission/run.py:1518` |
| `FUSION_DESEG_TOL` | `"4.0"` | `submission/run.py:1520` |
| `FUSION_ITERATE` | — | `submission/run.py:323` |
| `FUSION_LLM` | — *(4 处)* | `submission/run.py:323` |
| `FUSION_PINKEY_FIX` | `"1"` | `submission/run.py:1390` |
| `FUSION_SYM_YSHIFT` | `"0"` | `submission/run.py:1464` |
| `STAGE1_WIRE_COLOUR` | — | `submission/run.py:324` |
| `STAGE2_PINROT` | — | `submission/run.py:324` |
| `SUBMISSION_CONF` | `"0.25"` | `submission/run.py:969` |
| `VLM_BREAK_AFTER` | `"2"` | `submission/run.py:1691` |
| `VLM_REREAD_COOLDOWN` | `"25"` | `submission/run.py:1796` |

<a id="schematic_parser"></a>

## schematic_parser

| 开关 | 默认值 | 定义位置 |
|---|---|---|
| `OFFICIAL_BBOX_PX` | `"5.0"` | `schematic_parser/official.py:79` |
| `OFFICIAL_COLLAPSE` | `"0"` | `schematic_parser/official.py:90` |
| `OFFICIAL_PIN_BOX_ONLY` | `"0"` | `schematic_parser/official.py:105` |
| `OFFICIAL_PIN_PX` | `"5.0"` | `schematic_parser/official.py:83` |
