import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
from skimage import measure
from heapq import heappush, heappop

# A few images from MNIST to try tracing with model
Image=np.array(
      [[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0., 105., 227., 253., 253., 122.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0., 199., 253., 252., 252., 252., 252., 159.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0., 211., 252., 232., 152.,  73., 167., 252., 215.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,197., 252., 182.,   0.,   0.,   0.,   0., 235., 243.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,188., 252., 103.,   0.,   0.,   0.,   0., 235., 229.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,189., 253.,  86.,   0.,   0., 139., 190., 211.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,232., 252., 200., 201., 252., 252.,  84.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,213., 245., 252., 253., 252., 242.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,  84., 253., 252., 160.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0., 253., 252.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,  89., 255., 253.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,  80., 253., 189.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0., 179., 232.,  84.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0., 225., 252., 115.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,153., 252., 164.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  68.,245., 243.,  79.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 237.,245.,  82.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 148., 252.,169.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 106., 253., 196.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 228., 129.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.]])
Image2=np.array(
       [[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 0.,   0.,   0.,   0.,   0., 126., 136., 0.,   0., 166., 255.,247., 127.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  94.,154., 170., 253., 253., 253., 253., 0., 0., 172., 253., 242.,195.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 238., 253., 253.,253., 253., 253., 253., 253., 253., 0.,  0,  82.,  82.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 219., 253., 253.,253., 253., 253., 198., 182., 247., 241.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  80., 156., 107.,253., 253., 205.,   0.,   0.,   0., 154.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,154., 253.,  90.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,139., 253., 190.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0., 190., 253.,  70.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0., 241., 225., 0., 0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,  81., 240., 0., 253., 119.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0., 0., 253., 253., 150.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,  93., 252., 253., 187.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0., 249., 253., 249.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0., 130., 183., 253., 253., 207.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0., 148., 229., 253., 253., 253., 250., 182.,   0.,   0., 0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,114., 221., 253., 253., 253., 253., 201.,  78.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 213.,253., 253., 253., 253., 198.,  81.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0., 171., 219.,0., 253.,253., 253., 195.,  80.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0., 172., 226., 253., 253., 0., 253.,244., 133.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0., 136., 253., 253., 253., 212., 0., 132.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,   0.,   0.,   0.,   0.,   0.]]
       )
Image3=np.array(
      [[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 203., 229.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  95., 254., 215.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0., 154., 185., 185., 223., 253., 253., 133., 175., 255., 188.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,110., 253., 253., 253., 246., 161., 228., 253., 253., 254.,  92.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 128.,245., 253., 158., 137.,   0.,   0.,   0., 233., 253., 233.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 139.,254., 223.,   0.,   0.,   0.,   0., 170., 254., 244., 106.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,212., 253., 161.,   0.,   0., 178., 253., 236., 113.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,155., 253., 228.,  80., 223., 253., 253., 109.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,141., 253., 253., 253., 254., 253., 154.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,110., 253., 253., 253., 254., 179.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,171., 254., 254., 254., 179.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 171.,253., 253., 253., 253., 178.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 123., 254.,253., 203., 156., 253., 200.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  93., 253., 254.,121.,   0.,  93., 253., 158.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 239., 253.,  76.,0.,   0., 219., 253., 126.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0., 133., 254., 191.,   0.,0., 108., 234., 254., 106.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0., 132., 253., 190.,   0.,85., 253., 236., 154.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0., 153., 253., 169., 192.,253., 253.,  77.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0., 112., 253., 253., 254.,236., 129.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 118., 243., 191.,113.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.],
       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,0.,   0.,   0.,   0.,   0.,   0.]])

# Support Functions
def GenerateMask(starty, startx, dy, dx, radius, yy, xx):
    offset_y = dy * radius * 0.5
    offset_x = dx * radius * 0.5
    yc = starty + offset_y
    xc = startx + offset_x
    return (yy - yc)**2 + (xx - xc)**2 <= radius**2
def fast_flood_fill(mask, start_points, visited=None):
    h, w = mask.shape
    if visited is None:
        visited = np.zeros_like(mask, dtype=bool)
    region = []
    q = deque()
    for (sy, sx) in start_points:
        if 0 <= sy < h and 0 <= sx < w and mask[sy, sx]:
            q.append((sy, sx))
            mask[sy, sx] = False
            visited[sy, sx] = True
    if not q:
        return region
    while q:
        y, x = q.popleft()
        y0, y1 = max(0, y-1), min(h, y+2)
        x0, x1 = max(0, x-1), min(w, x+2)
        ys, xs = np.nonzero(mask[y0:y1, x0:x1] & ~visited[y0:y1, x0:x1])
        if len(ys):
            ys = ys + y0
            xs = xs + x0
            for ny, nx in zip(ys, xs):
                mask[ny, nx] = False
                visited[ny, nx] = True
                q.append((ny, nx))
                region.append((ny, nx))
    return region, visited
def fast_path_directions(paths, starty, startx):
    start = np.array([starty, startx], dtype=np.float32)
    means = np.empty((len(paths), 4), dtype=np.float32)
    for i, p in enumerate(paths):
        m = np.mean(p, axis=0)
        v = m - start
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        means[i][0:2] = v
        means[i][2:4]=starty,startx
    return means
def path_direction(path):
    y0, x0 = path[0]
    y1, x1 = path[-1]
    vec = np.array([y1 - y0, x1 - x0], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.array([0.0, 0.0])
    return vec / norm
def cosine_similarity(v1, v2):
    return np.dot(v1, v2)
def merge_paths(paths, angle_threshold=0.95):
    paths_sorted = sorted(paths, key=lambda p: len(p), reverse=True)
    merged_paths = []
    for path in paths_sorted:
        dir1 = path_direction(path)
        merged = False
        for i, mpath in enumerate(merged_paths):
            dir2 = path_direction(mpath)
            if cosine_similarity(dir1, dir2) > angle_threshold:
                # Same direction, merge coordinates
                all_coords = set(mpath + path)  # union of coordinates
                # Sort along main direction
                start = np.array([0,0])
                sorted_coords = sorted(all_coords, key=lambda c: np.dot(np.array(c)-start, dir1))
                merged_paths[i] = sorted_coords
                merged = True
                break
        if not merged:
            merged_paths.append(path)
    return merged_paths
def trace_paths_from_points(points, start):
    point_set = set(map(tuple, points))
    visited = set()
    paths = []
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    def get_neighbors(y, x):
        return [(y+dy, x+dx) for dy, dx in directions if (y+dy, x+dx) in point_set]
    queue = deque()
    for ny, nx in get_neighbors(*start):
        queue.append(((ny, nx), (ny-start[0], nx-start[1])))
    visited.add(start)
    while queue:
        (y, x), (dy, dx) = queue.popleft()
        path = [start]
        while (y, x) in point_set and (y, x) not in visited:
            path.append((y, x))
            visited.add((y, x))
            ny, nx = y + dy, x + dx
            if (ny, nx) not in point_set:
                break
            y, x = ny, nx
        if len(path) > 2:
            paths.append(path)
        for ny, nx in get_neighbors(y, x):
            if (ny, nx) in point_set and (ny, nx) not in visited:
                new_dir = (ny - y, nx - x)
                queue.append(((ny, nx), new_dir))
    return paths
def farthest_in_direction(coords, start, direction):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction) + 1e-9
    vecs = coords - start
    proj = np.dot(vecs, direction)
    forward_mask = proj > 0
    if not np.any(forward_mask):
        return None
    idx = np.argmax(proj * forward_mask)
    return coords[idx]
def filter_similar_directions(target_dir, directions, angle_threshold=0.95):
    filtered_directions = []
    for direction in directions:
        sim = cosine_similarity(target_dir, direction[:2])
        if sim <= angle_threshold:
            filtered_directions.append(direction)
    return np.array(filtered_directions)
def angle_between(v1, v2):
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    dot = np.dot(v1, v2)
    det = v1[0]*v2[1] - v1[1]*v2[0]
    return np.arctan2(det, dot)
def curvature_sign(dirs):
    if len(dirs) < 3:
        return 0
    angles = [angle_between(dirs[i], dirs[i+1]) for i in range(len(dirs)-1)]
    return np.sign(np.sum(angles))
def choose_next_direction(dirs, options, w_curv=0.2):
    cur_dir = dirs[-1]
    curv_sign = curvature_sign(dirs)
    best_dir, best_score = None, -np.inf
    for i, o in enumerate(options):
        ang = angle_between(cur_dir, o)
        smooth_score = np.cos(abs(ang))
        bias = w_curv * (curv_sign * np.sign(ang))
        score = smooth_score + bias
        if score > best_score:
            best_score, best_dir, index = score, o, i
    return best_dir, index
def slerp_2d(a, b, t):
    a = np.array(a)
    b = np.array(b)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    theta = np.arccos(dot)
    if np.isclose(theta, 0):
        return a
    elif np.isclose(theta, np.pi):
        result = (1 - t) * a + t * b
        return result / np.linalg.norm(result)
    sin_theta = np.sin(theta)
    factor_a = np.sin((1 - t) * theta) / sin_theta
    factor_b = np.sin(t * theta) / sin_theta
    blended = factor_a * a + factor_b * b
    return blended / np.linalg.norm(blended)
def expand_outward_find_new_pixel_with_bias(matrix, visited, bias_point):
    rows, cols = len(matrix), len(matrix[0])
    bias_y, bias_x = bias_point
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
    if not isinstance(visited, set):
        visited = set(visited)
    frontier = []
    explored = set()
    for y, x in visited:
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                if (ny, nx) not in visited and (ny, nx) not in explored:
                    explored.add((ny, nx))
                    dy_b, dx_b = ny - bias_y, nx - bias_x
                    heappush(frontier, (dy_b * dy_b + dx_b * dx_b, ny, nx))
    while frontier:
        _, y, x = heappop(frontier)
        if matrix[y][x] != 0 and (y, x) not in visited:
            return (y, x)
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                if (ny, nx) not in visited and (ny, nx) not in explored:
                    explored.add((ny, nx))
                    dy_b, dx_b = ny - bias_y, nx - bias_x
                    heappush(frontier, (dy_b * dy_b + dx_b * dx_b, ny, nx))
    return None

# Tracer Model
def TraceThatImage(image,Starty,Startx,steps,radius_dilation,yy,xx):
    start=time.time()
    starty,startx=Starty,Startx
    total=int(np.count_nonzero(image)*0.95)
    allreached=set()
    currentallreached=set()
    contours = measure.find_contours(image, level=0.5)
    cvimage=(image).astype(np.uint8)
    dy,dx=0,0
    while len(allreached)<total:
        Mask=GenerateMask(starty,startx,0,0,3*radius_dilation,yy,xx)
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        overlay = np.zeros((*Mask.shape, 4))
        overlay[Mask] = (*(1,0,0), 0.5)
        plt.imshow(overlay)
        plt.axis('off')
        plt.show()
        focus,visited=fast_flood_fill(Mask & (image != 0),[np.array([int(starty),int(startx)])])
        paths=fast_path_directions(merge_paths(trace_paths_from_points(focus,(starty,startx)),0.7),starty,startx)
        queue = deque(paths)
        allreached.update(focus)
        while queue:
            dy,dx,starty,startx = queue.popleft()
            sy,sx=starty,startx
            Mask=GenerateMask(starty,startx,dy,dx,3*radius_dilation,yy,xx)
            plt.figure(figsize=(6, 6))
            plt.imshow(image, cmap='gray')
            overlay = np.zeros((*Mask.shape, 4))
            overlay[Mask] = (*(1,0,0), 0.5)
            plt.imshow(overlay)
            plt.axis('off')
            plt.show()
            focus,visited=fast_flood_fill(Mask & (image != 0),[np.array([starty,startx],dtype=np.int8)])
            allreached.update(focus)
            allpaths=[]
            for i in range(steps):
                allpaths.append((dy,dx))
                try:
                    sy,sx=farthest_in_direction(focus,np.array([sy,sx]),np.array([dy,dx]))
                except TypeError:
                    break
                Mask=GenerateMask(sy,sx,dy,dx,3*radius_dilation,yy,xx)
                focus,visited=fast_flood_fill(Mask & (image != 0),[np.array([sy,sx])],visited)
                allreached.update(focus)
                if not focus:
                    break
                inverse=dy*-1,dx*-1
                paths=fast_path_directions(merge_paths(trace_paths_from_points([v for v in focus if v not in currentallreached],(sy,sx)),0.7),sy,sx)
                currentallreached.update(focus)
                if paths.size > 0:
                    paths=filter_similar_directions(inverse,paths)
                    comparisons=[path[:2] for path in paths]
                    reference=[path[:2] for path in allpaths]
                    UnitVector,target=choose_next_direction(reference,comparisons)
                    for i, path in enumerate(paths):
                        if i != target:
                            queue.append(path)
                    dy,dx=slerp_2d([dy,dx],UnitVector,0.95)
                else:
                    UnitVector=np.sum(focus-np.array([sy,sx]),axis=0,dtype='float64')
                    vectory,vectorx=UnitVector/np.linalg.norm(UnitVector)
                    dy,dx=slerp_2d([dy,dx],[vectory,vectorx],0.25)
                plt.figure(figsize=(6, 6))
                plt.imshow(image, cmap='gray')
                overlay = np.zeros((*Mask.shape, 4))
                overlay[Mask] = (*(1,0,0), 0.5)
                plt.imshow(overlay)
                plt.axis('off')
                plt.show()
                if len(allreached)>total:
                    break
            if len(allreached)>total:
                break
        if len(allreached)>total:
            break
        starty,startx=expand_outward_find_new_pixel_with_bias(image,allreached,(sy,sx))
        allreached.add((starty,startx))
    print(time.time()-start)

"""
You should see a red circular mask sliding across the image the goal was stimulate how humans trace across an image. The represents the fovea vision tracing across the image
"""
H,W=28,28
yy, xx = np.mgrid[0:H, 0:W]
starty,startx = 8,18
TraceThatImage(Image,starty,startx,10,1,yy,xx)