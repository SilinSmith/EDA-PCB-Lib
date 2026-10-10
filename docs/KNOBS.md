# 环境开关总表

> 本文件由 `python tools/dev/gen_knobs.py --write` **自动生成**，不要手改。

全部调参都走环境变量。假值词表（`0` / `off` / `false` / `none` / `no` 一律视为关）
统一实现在 `envflag.py`；PowerShell 里 `$env:X=''` 是**删除**变量而不是关功能，
要关某个开关请写具体假值，例如 `$env:STAGE2_PINNUM_TIGHT='0'`。

共 **391** 个开关。

## 目录

- [stage1](#stage1)（129）
- [stage2](#stage2)（101）
- [stage3](#stage3)（2）
- [fusion](#fusion)（114）
- [vlm](#vlm)（12）
- [textrec](#textrec)（13）
- [cvparts](#cvparts)（6）
- [pipeline](#pipeline)（11）
- [submission](#submission)（3）

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
| `STAGE1_BIGIC` | `"1"` | `stage1/pipeline.py:71` |
| `STAGE1_CROSS_COS` | `"-0.85"` | `stage1/labels.py:161` |
| `STAGE1_CROSS_DOT_NEED` | `"3"` | `stage1/labels.py:163` |
| `STAGE1_CROSS_DOT_R` | `"3.0"` | `stage1/labels.py:162` |
| `STAGE1_CROSS_SPLIT` | `"1"` | `stage1/labels.py:160` |
| `STAGE1_CROSS_TAGS` | `"1"` | `stage1/labels.py:165` |
| `STAGE1_CROSS_WIDE` | `"0"` | `stage1/labels.py:164` |
| `STAGE1_CROSS_WIDE_MAX` | `"96"` | `stage1/labels.py:166` |
| `STAGE1_CV_WIN` | `"off"` | `stage1/detect.py:134` |
| `STAGE1_DROP_TYPES` | `"pin,testpoint,mosfet_npn"` | `stage1/detect.py:97` |
| `STAGE1_GROUP_TOL` | `"1.5"` | `stage1/labels.py:47` |
| `STAGE1_MAX_PIXELS_HARD` | — | `stage1/pipeline.py:425` |
| `STAGE1_MERGE_TP` | `"0"` | `stage1/detect.py:120` |
| `STAGE1_NMS_IOU` | `"0.5"` | `stage1/detect.py:81` |
| `STAGE1_OCC_BIG` | `"60"` | `stage1/pipeline.py:252` |
| `STAGE1_OCC_LOCAL` | `"0"` | `stage1/wires.py:70` |
| `STAGE1_OCC_MODE` | `"all"` | `stage1/pipeline.py:246` |
| `STAGE1_OWN` | `"off"` | `stage1/detect.py:73` |
| `STAGE1_OWN_DEMOTE` | `"0.85"` | `stage1/detect.py:75` |
| `STAGE1_PINC` | `"1"` | `stage1/pipeline.py:81` |
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
| `STAGE1_PIN_ATTR` | `"nearest"` | `stage1/nets.py:67` |
| `STAGE1_PIN_ATTR_SLACK` | `"6"` | `stage1/nets.py:68` |
| `STAGE1_PIN_AXIS` | `"1"` | `stage1/pins.py:240` |
| `STAGE1_PIN_AXIS_RATIO` | `"1.2"` | `stage1/pins.py:242` |
| `STAGE1_PIN_CLUSTER_IC` | `"12.0"` | `stage1/pins.py:94` |
| `STAGE1_PIN_INK_TEXT` | `"0"` | `stage1/pins.py:201` |
| `STAGE1_PIN_INK_TEXT_PAD` | `"0.0"` | `stage1/pins.py:203` |
| `STAGE1_PIN_JOIN_DIST` | `"12"` | `stage1/nets.py:37` |
| `STAGE1_PIN_JOIN_FAR` | `"0"` | `stage1/nets.py:49` |
| `STAGE1_PIN_JOIN_SYNTH` | `"3"` | `stage1/nets.py:59` |
| `STAGE1_PIN_LEAD` | `"26"` | `stage1/pins.py:276` |
| `STAGE1_PIN_LEAD_FREE` | `"1"` | `stage1/pins.py:283` |
| `STAGE1_PIN_LEAD_INK` | `"0"` | `stage1/pins.py:287` |
| `STAGE1_PIN_LEAD_INK_FRAC` | `"0.6"` | `stage1/pins.py:289` |
| `STAGE1_PIN_LEAD_ORPHAN` | `"0"` | `stage1/pins.py:294` |
| `STAGE1_PIN_LEAD_ORPHAN_D` | `"12.0"` | `stage1/pins.py:296` |
| `STAGE1_PIN_LEAD_PERP` | `"3.0"` | `stage1/pins.py:277` |
| `STAGE1_PIN_LEAD_SRC` | `"end"` | `stage1/pins.py:280` |
| `STAGE1_PIN_MAJOR` | `"1"` | `stage1/pins.py:256` |
| `STAGE1_PIN_OUT_BOX` | `"12.0"` | `stage1/pins.py:55` |
| `STAGE1_PIN_STUB` | `"gapfill"` | `stage1/pins.py:137` |
| `STAGE1_PIN_STUB_GAPTOL` | `"4.0"` | `stage1/pins.py:160` |
| `STAGE1_PIN_STUB_MINLEN` | `"8"` | `stage1/pins.py:159` |
| `STAGE1_PIN_STUB_RATIO` | `"0.6"` | `stage1/pins.py:141` |
| `STAGE1_PIN_STUB_TEXT` | `"1"` | `stage1/pins.py:189` |
| `STAGE1_PIN_STUB_TEXT_PAD` | `"0.0"` | `stage1/pins.py:191` |
| `STAGE1_SPLIT_JUNCTIONS` | `"1"` | `stage1/wires.py:869` |
| `STAGE1_SPLIT_TOL` | `"1.5"` | `stage1/wires.py:871` |
| `STAGE1_WIRE_BRIDGE` | `"0"` | `stage1/wires.py:845` |
| `STAGE1_WIRE_BRIDGE_ANGLE` | `"30"` | `stage1/wires.py:850` |
| `STAGE1_WIRE_BRIDGE_DE` | `"22"` | `stage1/wires.py:852` |
| `STAGE1_WIRE_BRIDGE_FRAC` | `"0.9"` | `stage1/wires.py:849` |
| `STAGE1_WIRE_BRIDGE_INK` | `"0.5"` | `stage1/wires.py:851` |
| `STAGE1_WIRE_BRIDGE_MIN_SIDE` | `"24"` | `stage1/wires.py:853` |
| `STAGE1_WIRE_BRIDGE_NEAR` | `"8"` | `stage1/wires.py:848` |
| `STAGE1_WIRE_BRIDGE_THICK` | `"9"` | `stage1/wires.py:854` |
| `STAGE1_WIRE_CLOSE_THIN` | `"0"` | `stage1/wires.py:620` |
| `STAGE1_WIRE_COLOUR` | `"hint"` | `stage1/wires.py:513` |
| `STAGE1_WIRE_COS` | `"-0.75"` | `stage1/labels.py:130` |
| `STAGE1_WIRE_DILATE` | `"0"` | `stage1/wires.py:633` |
| `STAGE1_WIRE_EXT` | `"8.0"` | `stage1/labels.py:95` |
| `STAGE1_WIRE_EXT_NEAREST` | `"0"` | `stage1/labels.py:97` |
| `STAGE1_WIRE_EXT_PERP` | `"2.0"` | `stage1/labels.py:96` |
| `STAGE1_WIRE_HUE_MAXFAIL` | `"3"` | `stage1/wires.py:471` |
| `STAGE1_WIRE_HUE_MAXSIDE` | `"1024"` | `stage1/wires.py:489` |
| `STAGE1_WIRE_HUE_TIMEOUT` | `"30"` | `stage1/wires.py:470` |
| `STAGE1_WIRE_INK` | `"1"` | `stage1/labels.py:133` |
| `STAGE1_WIRE_INK_FRAC` | `"0.5"` | `stage1/labels.py:134` |
| `STAGE1_WIRE_INK_MODE` | `""` | `stage1/wires.py:629` |
| `STAGE1_WIRE_OVER` | `"12"` | `stage1/labels.py:191` |
| `STAGE1_WIRE_OVER_COS` | `"0.9"` | `stage1/labels.py:198` |
| `STAGE1_WIRE_OVER_CROSS` | `"0"` | `stage1/labels.py:197` |
| `STAGE1_WIRE_PERP` | `"1.5"` | `stage1/labels.py:129` |
| `STAGE1_WIRE_THIN_W` | `""` | `stage1/wires.py:617` |
| `STAGE1_WIRE_TOL` | `"1.5"` | `stage1/labels.py:94` |
| `STAGE1_WIRE_WELD` | `"16.0"` | `stage1/labels.py:128` |
| `VLM_TIMEOUT` | `"600"` | `stage1/wires.py:467` |

<a id="stage2"></a>

## stage2

| 开关 | 默认值 | 定义位置 |
|---|---|---|
| `STAGE1_MERGE_TP` | `"0"` | `stage2/assemble.py:108` |
| `STAGE2_BOTH_TOL` | `"0.65"` | `stage2/text.py:87` |
| `STAGE2_BOXIC_LIB` | `""` | `stage2/libpins.py:91` |
| `STAGE2_BOX_PIN_DROP_UNREAD` | `"mixed"` | `stage2/assemble.py:167` |
| `STAGE2_BOX_PIN_NEED_NAME` | `"2"` | `stage2/assemble.py:206` |
| `STAGE2_BOX_PIN_PARTIAL` | `"1"` | `stage2/assemble.py:141` |
| `STAGE2_CKPT` | `r"Y:\workspace\EDA\work\ocr_mix\runs\best.pt"` | `stage2/text.py:42` |
| `STAGE2_CONFORM_AUDIT` | `"1"` | `stage2/assign.py:415` |
| `STAGE2_CONFORM_AUDIT_MAX` | `"400"` | `stage2/assign.py:416` |
| `STAGE2_CONFORM_CONF` | `"0.50"` | `stage2/assign.py:403` |
| `STAGE2_CONFORM_GLUE_PEN` | `"0.85"` | `stage2/assign.py:405` |
| `STAGE2_CONFORM_KEY` | `"0"` | `stage2/assign.py:400` |
| `STAGE2_CONFORM_TYPE_PEN` | `"0.85"` | `stage2/assign.py:406` |
| `STAGE2_CPNT_LIB` | `""` | `stage2/libpins.py:90` |
| `STAGE2_DBG_CID` | — | `stage2/assign.py:102` |
| `STAGE2_DBG_PINS` | — *(4 处)* | `stage2/assemble.py:964` |
| `STAGE2_DEMOTE_HARD` | `"0.05"` | `stage2/structmatch.py:35` |
| `STAGE2_DESIG_MAXW` | `"6.0"` | `stage2/assign.py:197` |
| `STAGE2_DROP_UNKEYED` | `"1"` | `stage2/assemble.py:59` |
| `STAGE2_EDGE_ANG` | `"6.0"` | `stage2/assemble.py:350` |
| `STAGE2_EDGE_JOIN` | `"40.0"` | `stage2/assemble.py:349` |
| `STAGE2_EDGE_MERGE` | `"0"` | `stage2/assemble.py:339` |
| `STAGE2_EDGE_SNAP` | `"10.0"` *(2 处)* | `stage2/assemble.py:348` |
| `STAGE2_EDGE_SNAP_ANG` | `"45.0"` | `stage2/assemble.py:371` |
| `STAGE2_FREE_STEAL` | `"1"` | `stage2/assign.py:255` |
| `STAGE2_GLOBAL_MATCH` | `"0"` | `stage2/assign.py:986` |
| `STAGE2_GROW_MIN` | `"0.9"` | `stage2/run.py:31` |
| `STAGE2_GROW_PER_COMP` | `"4"` | `stage2/run.py:33` |
| `STAGE2_LIB_PINS` | `"1"` *(2 处)* | `stage2/assemble.py:232` |
| `STAGE2_LIB_PINS_DBG` | `"0"` | `stage2/assemble.py:234` |
| `STAGE2_LIB_PINS_FUZZY` | `"0.85"` | `stage2/libpins.py:199` |
| `STAGE2_MIN_EDGE_LEN` | `"20"` | `stage2/assemble.py:313` |
| `STAGE2_MULTI_H` | `"0.43"` | `stage2/text.py:95` |
| `STAGE2_MULTI_W` | `"0.15"` | `stage2/text.py:94` |
| `STAGE2_NETLABEL_MIN_H` | `"9"` | `stage2/assign.py:337` |
| `STAGE2_NETLABEL_MIN_W` | `"12"` | `stage2/assign.py:336` |
| `STAGE2_NET_DROP_PWRONLY` | `"1"` | `stage2/assemble.py:276` |
| `STAGE2_NET_DROP_UNKEYED` | `"0"` | `stage2/assemble.py:265` |
| `STAGE2_NODELETE` | `"1"` | `stage2/structmatch.py:33` |
| `STAGE2_OWN_ASSIGN` | `"0"` | `stage2/assign.py:87` |
| `STAGE2_OWN_BBOX` | `"1"` | `stage2/assign.py:68` |
| `STAGE2_OWN_MATCH` | `"0"` | `stage2/assign.py:98` |
| `STAGE2_PAD_FRAC` | `"0.0"` | `stage2/text.py:70` |
| `STAGE2_PAD_FRAC_Y` | `"0.20"` | `stage2/text.py:79` |
| `STAGE2_PAD_MAX` | `"99.0"` | `stage2/text.py:71` |
| `STAGE2_PAD_MIN` | `"2.0"` | `stage2/text.py:69` |
| `STAGE2_PAD_MIN_Y` | `"3.0"` | `stage2/text.py:80` |
| `STAGE2_PINCOL` | `"0"` | `stage2/assign.py:242` |
| `STAGE2_PINCOL_COVER` | `"0.5"` | `stage2/assign.py:243` |
| `STAGE2_PINCOL_WIN` | `"1"` | `stage2/run.py:37` |
| `STAGE2_PINCOL_WIN_KEYS` | `"out_win"` | `stage2/run.py:40` |
| `STAGE2_PINNAME_MAXW` | `"6.0"` | `stage2/assign.py:163` |
| `STAGE2_PINNAME_OUTSIDE` | `"0"` | `stage2/assign.py:187` |
| `STAGE2_PINNAME_RESCUE` | `"0"` | `stage2/pinname.py:56` |
| `STAGE2_PINNAME_RESCUE_AGREE` | `"1"` | `stage2/pinname.py:60` |
| `STAGE2_PINNAME_RESCUE_CONF` | `"0.70"` | `stage2/pinname.py:58` |
| `STAGE2_PINNAME_RESCUE_DEPTH` | `"52.0"` | `stage2/pinname.py:63` |
| `STAGE2_PINNAME_RESCUE_GRAY` | `"170"` | `stage2/pinname.py:65` |
| `STAGE2_PINNAME_RESCUE_INSET` | `"4.0"` | `stage2/pinname.py:62` |
| `STAGE2_PINNAME_RESCUE_SAT` | `"70"` | `stage2/pinname.py:66` |
| `STAGE2_PINNUM_DECAY` | `"10"` | `stage2/assign.py:132` |
| `STAGE2_PINNUM_DISTW` | `"0.005"` | `stage2/assign.py:119` |
| `STAGE2_PINNUM_DUP` | `"0"` | `stage2/assign.py:144` |
| `STAGE2_PINNUM_NEAR` | `"60"` | `stage2/assign.py:115` |
| `STAGE2_PINNUM_PAD_FRAC_Y` | `"0.0"` | `stage2/text.py:125` |
| `STAGE2_PINNUM_STEAL` | `"0"` | `stage2/assign.py:159` |
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
| `STAGE2_PIN_ALONG` | `"1"` | `stage2/assign.py:207` |
| `STAGE2_PIN_COVER` | `"0.15"` | `stage2/assign.py:211` |
| `STAGE2_PIN_RESIDUAL` | `"0"` | `stage2/assign.py:219` |
| `STAGE2_PIN_RESIDUAL_MIN` | `"0"` | `stage2/assign.py:220` |
| `STAGE2_PIN_STAMP` | `"1"` | `stage2/assemble.py:506` |
| `STAGE2_PREFIX` | `"strict"` | `stage2/assign.py:38` |
| `STAGE2_PREFIX_SKEL` | `"0"` | `stage2/assign.py:48` |
| `STAGE2_RAIL_ATTACH` | `"1"` | `stage2/assemble.py:1243` |
| `STAGE2_STAMP_ONE_SIDE` | `"1"` | `stage2/assemble.py:522` |
| `STAGE2_STEAL` | `"80"` | `stage2/assign.py:253` |
| `STAGE2_SYM_MERGE` | `"0"` | `stage2/assemble.py:299` |
| `STAGE2_TIGHT_ROLES` | `"pin_number,pin_name"` | `stage2/text.py:129` |
| `STAGE2_TP_CV_SUBTYPES` | `"ring3"` | `stage2/assemble.py:110` |
| `STAGE2_TP_LEDGER` | `"0"` | `stage2/assemble.py:125` |
| `STAGE2_TP_TRUST_EVIDENCE` | `"1"` | `stage2/assemble.py:122` |
| `STAGE2_VALUE_DISTW` | `"0.035"` | `stage2/assign.py:318` |
| `STAGE2_VALUE_NETNAME` | `"0"` | `stage2/assign.py:307` |
| `STAGE2_VALUE_STEAL` | `"40"` | `stage2/assign.py:264` |
| `STAGE2_VALUE_TRIM` | `"0"` | `stage2/assign.py:294` |
| `STAGE2_VALUE_UNIT` | `"1"` | `stage2/assign.py:282` |

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
| `FUSION_ASSIGNED_DIR` | `str(RUN_DIR / "stage2" / "_assigned"` | `fusion/run.py:188` |
| `FUSION_AUDIT` | `"1"` | `fusion/audit.py:76` |
| `FUSION_AUDIT_FULLW` | `"880"` | `fusion/audit.py:77` |
| `FUSION_AUDIT_KINDS` | `"net_short,net_rail_merge,key_duplicate,key_type_mismatch,va` | `fusion/audit.py:82` |
| `FUSION_AUDIT_MAX` | `"16"` | `fusion/audit.py:69` |
| `FUSION_AUDIT_MAXH` | `"1400"` | `fusion/audit.py:78` |
| `FUSION_AUDIT_REREAD` | `"1"` | `fusion/audit.py:74` |
| `FUSION_AUDIT_SCORE` | `"1.30"` | `fusion/audit.py:68` |
| `FUSION_AXIS` | `"1"` | `fusion/axis.py:87` |
| `FUSION_AXIS_VERT` | `"0"` | `fusion/axis.py:155` |
| `FUSION_BASE` | `"ours"` | `fusion/run.py:129` |
| `FUSION_BOXSCAN` | `"0"` | `fusion/claims.py:129` |
| `FUSION_BOX_PIN_PRIOR` | `"0"` | `fusion/priors.py:135` |
| `FUSION_CASES` | `""` | `fusion/run.py:371` |
| `FUSION_CONV_EXACT` | `"1"` | `fusion/run.py:91` |
| `FUSION_CROPFIX` | `"1"` | `fusion/cropfix.py:61` |
| `FUSION_CROPFIX_LADDER` | `"0.5,1.0,2.0,4.0"` | `fusion/cropfix.py:54` |
| `FUSION_CROPFIX_MAX` | `"24"` | `fusion/cropfix.py:57` |
| `FUSION_CROPFIX_ROUNDS` | `"2"` | `fusion/cropfix.py:56` |
| `FUSION_CROPFIX_SCALE` | `os.environ.get("VLM_SCALE", "4"` | `fusion/cropfix.py:50` |
| `FUSION_CROPFIX_SCORE` | `"1.25"` | `fusion/cropfix.py:58` |
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
| `FUSION_LLM` | `"1"` | `fusion/run.py:87` |
| `FUSION_LLM_SCORE` | `"1.5"` | `fusion/llm_judge.py:45` |
| `FUSION_MAX_CANDS` | `"8"` | `fusion/solve.py:73` |
| `FUSION_MAX_OPEN` | `"24"` | `fusion/solve.py:49` |
| `FUSION_MERGE_PIN_MIN` | `"8"` | `fusion/run.py:158` |
| `FUSION_MIN_NET_PINS` | `"2"` | `fusion/plausible.py:57` |
| `FUSION_OPEN_FREEFORM` | `"1"` | `fusion/solve.py:71` |
| `FUSION_OPEN_FREEFORM_MAX` | `"8"` | `fusion/solve.py:60` |
| `FUSION_OPEN_MARGIN` | `"0.10"` | `fusion/solve.py:47` |
| `FUSION_OUT_DIR` | `str(ROOT / "work" / "_fusion_out"` | `fusion/run.py:68` |
| `FUSION_READ_DROP1` | `"0.40"` | `fusion/priors.py:426` |
| `FUSION_READ_DROP1O` | `"0.12"` | `fusion/priors.py:427` |
| `FUSION_READ_DROPM` | `"0.12"` | `fusion/priors.py:428` |
| `FUSION_READ_EXACT` | `"1.00"` | `fusion/priors.py:425` |
| `FUSION_READ_FAR` | `"0.01"` | `fusion/priors.py:431` |
| `FUSION_READ_INS` | `"0.30"` | `fusion/priors.py:429` |
| `FUSION_READ_ROW_OVERLAP` | `"0.5"` | `fusion/priors.py:282` |
| `FUSION_READ_SUB` | `"0.05"` | `fusion/priors.py:430` |
| `FUSION_REGION_MIN` | `"120"` | `fusion/claims.py:92` |
| `FUSION_REGION_PAD` | `"24"` | `fusion/claims.py:90` |
| `FUSION_REGION_TOPK` | `"1"` | `fusion/claims.py:134` |
| `FUSION_REPAIR_PENALTY` | `"0.85"` | `fusion/priors.py:223` |
| `FUSION_RESCAN_CROP` | `"window"` | `fusion/rescan.py:42` |
| `FUSION_RESCAN_MAX` | `"12"` | `fusion/rescan.py:40` |
| `FUSION_RESCAN_SCORE` | `"1.2"` | `fusion/rescan.py:38` |
| `FUSION_RESCAN_VALUE` | `"0"` | `fusion/rescan.py:98` |
| `FUSION_ROUNDS` | `"2"` | `fusion/run.py:75` |
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
| `FUSION_STAGE1_DIR` | `str(RUN_DIR / "stage1"` *(2 处)* | `fusion/run.py:186` |
| `FUSION_TOP_WINDOWS` | `"3"` | `fusion/priors.py:225` |
| `FUSION_VALUE_MIN_N` | `"4"` | `fusion/plausible.py:55` |
| `FUSION_VALUE_RATIO` | `"4.0"` | `fusion/plausible.py:53` |
| `FUSION_VLM_DIR` | `r"Y:\workspace\EDA\work\_fusion_vlm"` *(2 处)* | `fusion/audit.py:556` |
| `FUSION_VLM_PIN_NAME` | `"1"` | `fusion/claims.py:83` |
| `FUSION_VLM_PIN_NUM` | `"1"` | `fusion/claims.py:82` |
| `FUSION_VLM_PIN_SCORE` | `"1.15"` | `fusion/claims.py:74` |
| `FUSION_VLM_SCORE` | `"1.15"` | `fusion/claims.py:71` |
| `FUSION_WORKERS` | `"1"` | `fusion/run.py:393` |
| `VLM_TIMEOUT` | `"600"` *(3 处)* | `fusion/audit.py:75` |

<a id="vlm"></a>

## vlm

| 开关 | 默认值 | 定义位置 |
|---|---|---|
| `LLM_API_KEY` | `""` | `vlm/client.py:33` |
| `LLM_BASE_URL` | `"http://127.0.0.1:1234/v1"` | `vlm/client.py:32` |
| `LLM_MODEL` | `"qwen3.5-9b"` | `vlm/client.py:34` |
| `VLM_MAX_TOKENS` | `"2048"` | `vlm/client.py:36` |
| `VLM_PIN_CHANNEL` | `"0"` | `vlm/enrich.py:819` |
| `VLM_PIN_COLS` | `"4"` | `vlm/enrich.py:828` |
| `VLM_PIN_MERGE` | `"0"` | `vlm/enrich.py:785` |
| `VLM_PIN_PER` | `"8"` | `vlm/enrich.py:827` |
| `VLM_PIN_REBUILD` | `"1"` | `vlm/enrich.py:793` |
| `VLM_REASONING` | `"none"` | `vlm/client.py:37` |
| `VLM_SCALE` | `"4"` | `vlm/client.py:35` |
| `VLM_TIMEOUT` | `"600"` | `vlm/client.py:38` |

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
| `CV_TP_DOT` | `"1"` | `cvparts/testpoint.py:91` |
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
| `STAGE1_WIRE_COLOUR` | `"hint"` | `pipeline/pskel.py:272` |
| `STAGE1_WIRE_DEBUG` | — | `pipeline/pskel.py:338` |
| `STAGE1_WIRE_EDGE` | `"0"` | `pipeline/pskel.py:515` |
| `STAGE1_WIRE_EDGE_MIN` | `"10.0"` | `pipeline/pskel.py:586` |
| `STAGE1_WIRE_EDGE_NEAR` | `"2"` | `pipeline/pskel.py:599` |
| `STAGE1_WIRE_EDGE_REL` | `"0.55"` | `pipeline/pskel.py:582` |
| `STAGE1_WIRE_UNCONF` | `"veto"` | `pipeline/pskel.py:552` |
| `STAGE1_WIRE_UNCONF_NEAR` | `"3"` | `pipeline/pskel.py:566` |
| `WINDIR` | `r'C:\Windows'` | `pipeline/pglyph.py:30` |

<a id="submission"></a>

## submission

| 开关 | 默认值 | 定义位置 |
|---|---|---|
| `EDA_CASE_WORKERS` | `""` | `submission/run.py:293` |
| `EDA_WALL_BUDGET_S` | `"0"` | `submission/run.py:442` |
| `FUSION_LLM` | `"1"` *(3 处)* | `submission/run.py:783` |
