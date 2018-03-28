import sys
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import *

from util import read_scaled_img_with_border

def smooth(pts):
    return (pts[1:-1] + pts[:-2] + pts[2:])/3

FRAME_BORDER_SIZE = 100
FRAME_SCALE = 2.0

trail_len = 20
alphas = np.linspace(0.25, 1, trail_len)

capture = cv2.VideoCapture(sys.argv[1])
if not capture.isOpened():
    raise IOError("Failed to open video: '%s'" % sys.argv[1])

ellipses = pickle.load(open(sys.argv[2], "rb"))

frames = []
frame_good, frame = read_scaled_img_with_border(capture, FRAME_BORDER_SIZE, FRAME_SCALE);
for i in range(len(ellipses)):
    cv2.ellipse(frame, tuple(ellipses[i]), (0,255,0), 2)
    cv2.circle(frame, tuple(int(f) for f in ellipses[i][0]), 6, (0,0,255), -1)
    frames.append(frame[:,:,::-1])
    frame_good, frame = read_scaled_img_with_border(capture, FRAME_BORDER_SIZE, FRAME_SCALE);
    #pass

frame_dim = frames[0].shape[:2]

centers = np.array([el[0] for el in ellipses])
centers = (centers[1:-1] + centers[:-2] + centers[2:]) / 3.
frames = frames[1:-1]
print(len(frames))
print(len(centers))

mean = centers.mean(axis=0)
dists = np.linalg.norm(centers - mean[np.newaxis], axis=-1)
xi = np.linspace(0, len(dists)-1, 2*len(dists))
di = np.interp(xi, range(len(dists)), dists)
di = smooth(smooth(di))
xi = smooth(smooth(xi))
vi = np.diff(di)
zc = np.diff(np.sign(vi)).nonzero()[0]+1

plt.plot(xi, di, 'b')
plt.plot(np.round(xi[zc]).astype(np.int32), di[zc], 'r*')
plt.savefig("zero_crossings.png")
plt.close()
out = open("zcs.txt", "w")
out.write("\n".join([str(x) for x in np.round(xi[zc]).astype(np.int32)]))

for fr in range(len(frames)):
    plt.figure(figsize=(9,9))
    plt.subplot2grid((2,2), (0,0), colspan=2)
    plt.imshow(frames[fr])
    plt.axis("off")

    plt.subplot2grid((2,2), (1,0))
    plt.xlim(0, frame_dim[1])
    plt.ylim(0, frame_dim[0])
    plt.plot(centers[fr][0], centers[fr][1], 'bo')
    for j in range(max(0, fr-trail_len), fr):
        p0 = centers[j]; p1 = centers[j+1]
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]), 'r-', alpha=alphas[j-fr])
    plt.xlabel("X position (px)")
    plt.ylabel("Y position (px)")

    plt.subplot2grid((2,2), (1,1))
    plt.plot(range(fr+1), dists[:fr+1])
    plt.xlabel("Frame #")
    plt.ylabel("Distance from Center of Motion (px)")
    plt.xlim(0, len(frames))
    plt.ylim(0, dists.max())
    plt.plot([fr], dists[fr], 'r.')
    plt.savefig("out/%d.png" % fr, bbox_inches="tight")
    plt.close()
    img = imread("out/%d.png" % fr)
    imsave("out/%d.png" % fr, img[:img.shape[0] - (img.shape[0]%2), :img.shape[1] - (img.shape[1]%2)])
    print(fr)
