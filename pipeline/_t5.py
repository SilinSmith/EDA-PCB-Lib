import sys, os, cv2, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
p = r'Y:\workspace\EDA\合成原理图demo\images\synth_0001.png'
print('exists', os.path.exists(p), 'abspath', os.path.abspath(p))
im = cv2.imread(p, cv2.IMREAD_COLOR)
print('imread', None if im is None else im.shape)
print('imread_unicode', None if cv2.imread(p, cv2.IMREAD_UNCHANGED) is None else cv2.imread(p, cv2.IMREAD_UNCHANGED).shape)
buf = np.fromfile(p, dtype=np.uint8)
im2 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
print('imdecode', None if im2 is None else im2.shape)
try:
    import PIL.Image as I
    print('PIL', I.open(p).size)
except Exception as e:
    print('PIL err', e)
