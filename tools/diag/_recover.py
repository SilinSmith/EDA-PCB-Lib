# 反演：PS 以 cp936 读 UTF-8 文件 -> 得到 mojibake 字符串 M -> 又按 UTF-8 写回。
# 所以 M = decode(utf-8, C)，原字节 O ≈ encode(gbk, M)。
src = r"eda-pcb\fusion\run.py"
C = open(src, "rb").read()
M = C.decode("utf-8")
print("chars:", len(M), " 含替换符 U+FFFD:", M.count("\ufffd"), " 含 '?':", M.count("?"))
try:
    O = M.encode("gbk")
    print("gbk 编码成功, bytes:", len(O))
except Exception as e:
    print("gbk 编码失败:", e)
    O = M.encode("gbk", errors="replace")
open(r"work\tools\diag\_run_recovered.py", "wb").write(O)
T = O.decode("utf-8", errors="replace")
print("恢复后 UTF-8 解码替换符数:", T.count("\ufffd"))
open(r"work\tools\diag\_run_recovered.view.txt", "w", encoding="utf-8").write(T)
import ast
try:
    ast.parse(T); print("恢复后可直接 parse: OK")
except SyntaxError as e:
    print("parse 失败 line", e.lineno, ":", e.msg)
    print("   >>>", (T.splitlines()[e.lineno-1] if e.lineno and e.lineno <= len(T.splitlines()) else "")[:120])
