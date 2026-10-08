# -*- coding: utf-8 -*-
"""把 workflow.html 里 kind='new'（★ 要新写）的节点的 note + bullets 全文倒出来，
用于逐条核对「还有哪些没写」。

用法：
  python work/tools/diag/_wfnew.py                # 只打标题与长度
  python work/tools/diag/_wfnew.py --all          # 全部 new 节点打全文
  python work/tools/diag/_wfnew.py <id> ...       # 指定节点打全文
  python work/tools/diag/_wfnew.py --file <路径>  # 全文写到文件
"""
import io
import json
import re
import sys

SRC = r'Y:\workspace\EDA\work\flow\workflow.html'


def load(path=SRC):
    t = io.open(path, encoding='utf-8').read()
    m = re.search(r'application/json[^>]*>(.*?)</script>', t, re.S)
    return json.loads(m.group(1))


def render(n):
    L = []
    L.append('=' * 78)
    L.append('%-12s kind=%s group=%s' % (n['id'], n['kind'], n['group']))
    L.append('标题：%s' % n.get('title'))
    if n.get('subtitle'):
        L.append('副标题：%s' % n['subtitle'])
    if n.get('note'):
        L.append('--- note (%d 字) ---' % len(n['note']))
        L.append(n['note'])
    bl = n.get('bullets') or []
    if bl:
        L.append('--- bullets (%d 条) ---' % len(bl))
        for i, b in enumerate(bl, 1):
            L.append('  [%02d] %s' % (i, b))
    if n.get('params'):
        L.append('--- params (%d) ---' % len(n['params']))
        for p in n['params']:
            L.append('  %s' % (p,))
    L.append('')
    return '\n'.join(L)


def main():
    d = load()
    args = sys.argv[1:]
    news = [n for n in d['nodes'] if n.get('kind') == 'new']
    if args and args[0] == '--file':
        out = '\n'.join(render(n) for n in news)
        p = args[1] if len(args) > 1 else r'Y:\workspace\EDA\work\flow\_wfnew.txt'
        io.open(p, 'w', encoding='utf-8').write(out)
        print('wrote %s (%d chars, %d nodes)' % (p, len(out), len(news)))
        return
    if not args:
        for n in news:
            print('%-12s note=%5d bullets=%2d params=%2d  %s'
                  % (n['id'], len(n.get('note', '')), len(n.get('bullets') or []),
                     len(n.get('params') or []), (n.get('title') or '')[:44]))
        print('\n合计 new 节点：%d' % len(news))
        return
    if args == ['--all']:
        for n in news:
            print(render(n))
        return
    byid = {n['id']: n for n in d['nodes']}
    for a in args:
        print(render(byid[a]) if a in byid else '!! 没有这个节点：%s' % a)


if __name__ == '__main__':
    main()
