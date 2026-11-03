"""同槽双通道联合断言：脚号 PP-OCR ∧ 脚名 PP-OCR。

问题：`pin_number_reading` 与 `pin_name_reading` 是两条**互相独立**的外部通道
（不同裁片、不同判据、各自 ≥.90 的接受门），但它们的读数进 `assign` 之后是按
**槽**绑定角色的：一颗箱体件的号落到 idx_a、名字落到 idx_b 时，两个槽各自缺一半，
`assemble.py` 的「迟到 need-name 闸门」就把缺名字的那一槽整条丢掉 ⇒ 两条通道都
对同一根脚给了证据，却谁都发不出去（`case11 U1` 就是这样丢 6 根脚）。

本模块把「两条通道都对同一个 `(component_id, pin_index)` 给出非空读数」当成一次
外部断言：这种槽用缺的那一半补齐。判据里**没有任何阈值/半径/距离** —— 一致性就是
「两侧对同一 pin_index 都有读数」；置信度门沿用管线自己的 `min_conf`。

只补不删、不新增槽、不覆盖已有的 `number`/`name` ⇒ 开关关掉时下游逐位不变。
"""

# 通道名（`run.py` 的 `PIN_READERS` 注册表里那两个名字）→ (reads 角色, 槽位字段)
CHANNELS = (
    ("pin_number_repair", "pin_number", "number"),
    ("pin_name_ocr", "pin_name", "name"),
)


def _accepted(records, role, min_conf):
    """`(component_id, pin_index)` → (text, conf, bbox)：只留被接受的读数。

    同一 `pin_index` 可能有多条候选（不同裁片/方向），取置信度最高的一条 ——
    这里不做「谁的几何更好」的判断，那是 `assign` 的活。
    """
    out = {}
    for rec in records or []:
        target = rec.get("pin_target") or {}
        cid, idx = target.get("component_id"), target.get("pin_index")
        if cid is None or idx is None:
            continue
        text, conf = (rec.get("reads") or {}).get(role) or ("", 0.0)
        text = str(text or "").strip()
        if not text or float(conf) < min_conf:
            continue
        key = (str(cid), int(idx))
        old = out.get(key)
        if old is None or float(conf) > old[1]:
            out[key] = (text, float(conf), rec.get("glyph_bbox") or rec.get("bbox"))
    return out


def _claimed(pins, skip_idx, number):
    """同一个元件里，是否已有**别的槽**用这个号、并且带名字（⇒ 已经在发 `pin_N`）。"""
    for j, rec in (pins or {}).items():
        if int(j) == skip_idx or not rec or rec.get("number") is None:
            continue
        if (rec["number"].get("text") or "").strip() != number:
            continue
        if rec.get("name") is not None:
            return True
    return False


def join(assigned, channel_reads, types=None, *, min_conf=0.0):
    """把两条通道**同时**接受的槽补上缺的那一半；返回计数 dict 供日志。

    `assigned` 是 `assign()` 的产物（原地修改，`comps[cid]["pins"][idx]`），
    `channel_reads` = {通道名: 该读取器返回的 `List[dict]`}，
    `types` = {component_id: 类型}（只补 `box`：其它类型的引脚名本来就有
    `nm = r["nm"] or num` 的兜底，补名反而会盖掉兜底）。

    ★ 本函数**只做「增脚」**，不做「换身份」。两条结构判据（都不是新阈值）：

    1. 槽已经有名字 ⇒ 跳过。它有名字就等于**已经发得出去**（`assemble` 的
       need-name 闸门只卡「没名字」的箱体脚），这时补号只会把它换个号码重发，
       实测是五五开：`case1 U1.pin_2` 由 `VIN` 变 `LED1` 是真号（赚），但
       `case5 U3.pin_4` 由 `TAB` 变 `T△R`、`case7 U1.pin_1` 变 `GNV` 都是亏。
    2. 断言出的号若**已被同一元件里另一个带名字的槽占着**，跳过：那一槽本来就会
       发出 `pin_N`，我们再补一个只会让先到者（`PIN_DUPNUM` 保留编号小的槽）
       换几何 —— `case4 U2.pin_4` 的点因此挪了 3px、`case5` 三处同号抢位都是亏。
       反过来，占着号但**没名字**的槽本来就会被丢掉（`case11 U1` 的 idx1 就是
       `(8, 无名字)`），此时断言补的是唯一能发出去的那一份 ⇒ 不跳过。

    ★ 2026-10-31 实测：把判据 1 删掉、判据 2 改成「占号即跳过（不看名字）」（离线模型
    预测 +0.128 判题机分）在真实管线上是 **local −0.1274 / judge −0.0104**（case7
    −0.77、case11 −0.88，都是「补出的号触发 `PIN_DUPNUM` 让位 / 重编号」把已发对的脚
    挤走），故**保持原判据**。见 `docs/RESULTS.md` §5.519。
    """
    comps = assigned.get("comps") or {}
    maps = {role: _accepted(channel_reads.get(name), role, min_conf)
            for name, role, _f in CHANNELS}
    stats = {"assert": 0, "filled_number": 0, "filled_name": 0,
             "skip_named": 0, "skip_claimed": 0, "skipped_type": 0,
             "skipped_no_slot": 0}
    shared = sorted(set(maps["pin_number"]) & set(maps["pin_name"]))
    for cid, idx in shared:
        stats["assert"] += 1
        if types is not None and (types.get(cid) or "").strip().lower() != "box":
            stats["skipped_type"] += 1
            continue
        pins = (comps.get(cid) or {}).get("pins") or {}
        pin = pins.get(idx)
        if pin is None:                       # 槽不存在 ⇒ 不新增槽
            stats["skipped_no_slot"] += 1
            continue
        if pin.get("name") is not None:       # 判据 1：已在发，不换身份
            stats["skip_named"] += 1
            continue
        number = maps["pin_number"][(cid, idx)][0]
        if _claimed(pins, idx, number):       # 判据 2：不跟已发的同号脚抢几何
            stats["skip_claimed"] += 1
            continue
        for name, role, field in CHANNELS:
            if pin.get(field) is not None:
                continue
            text, conf, box = maps[role][(cid, idx)]
            rec = {"text": text, "conf": conf, "box": list(box or []), "d": 0.0}
            rec["pin_ocr_join"] = role
            pin[field] = rec
            stats["filled_" + field] += 1
    return stats

