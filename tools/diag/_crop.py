from PIL import Image
im = Image.open(r"work\_submit_in220\1.png").convert("RGB")
W,H = im.size
print("size", W, H)
box = (1340, 180, min(W,1478), 420)
c = im.crop(box).resize(((box[2]-box[0])*4, (box[3]-box[1])*4), Image.NEAREST)
c.save(r"work\preview\_c1right.png")
box2 = (700, 60, 960, 480)
c2 = im.crop(box2).resize(((box2[2]-box2[0])*3, (box2[3]-box2[1])*3), Image.NEAREST)
c2.save(r"work\preview\_c1rail.png")
print("ok")
