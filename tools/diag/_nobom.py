import io
p = r"eda-pcb\fusion\run.py"
raw = open(p, "rb").read()
if raw.startswith(b"\xef\xbb\xbf"):
    open(p, "wb").write(raw[3:])
    print("已去掉 BOM")
else:
    print("没有 BOM")
