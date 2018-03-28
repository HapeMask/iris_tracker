#!/usr/bin/python

import pickle
import sys
import numpy as np
import cv2

from iris_track import update_ellipse as _upd_el
from util import read_scaled_img_with_border, pick_ellipse
from kalman import KalmanFilter

FRAME_BORDER_SIZE = 100
FRAME_SCALE = 2.0
TRACKER_LINE_LEN = 20
N_TRACKER_LINES = 40

def update_ellipse(el, image, allow_approx = True, line_len=TRACKER_LINE_LEN, n_lines=N_TRACKER_LINES):
    args = el[0] + el[1] + (el[2], image, allow_approx, line_len, n_lines)
    success, el = _upd_el(*args)
    return success, list(el)

display = True

capture = cv2.VideoCapture(sys.argv[1])
if not capture.isOpened():
    raise IOError("Failed to open video: '%s'" % sys.argv[1])
fps = capture.get(cv2.CAP_PROP_FPS)

frame_good, image = read_scaled_img_with_border(capture, FRAME_BORDER_SIZE, FRAME_SCALE)
#el = pick_ellipse(image)
#print("el =", el.__repr__())
#exit()
#el = ((305.01300048828125, 287.8521423339844), (118.04708099365234, 122.06232452392578), 95.16199493408203) # P1 Pre v0
#el = ((234.84469604492188, 238.71983337402344), (108.19189453125, 119.11138916015625), 75.62329864501953) # P1 Post v0
#el = ((639.935546875, 613.4555053710938), (238.23541259765625, 257.32257080078125), 74.572265625) # P2 Pre v0
#el = ((639.14794921875, 592.5144653320312), (312.1429138183594, 322.047119140625), 170.886962890625) # P2 Post v0
#el = ((493.0901184082031, 462.36431884765625), (206.107666015625, 213.04690551757812), 86.48485565185547) # P3 Pre v0
#el = ((448.2835998535156, 446.7442932128906), (208.44630432128906, 218.11141967773438), 22.631765365600586) # P3 Post v0
#el = ((420.3846740722656, 391.48138427734375), (198.2941131591797, 201.3717498779297), 66.53734588623047) # P4 Pre v0
#el = ((375.72698974609375, 367.2718200683594), (189.04757690429688, 196.62730407714844), 62.1186637878418) # P4 Post v2
#el = ((538.9722290039062, 495.59918212890625), (298.58258056640625, 305.5045166015625), 108.21113586425781) # P5 Pre v0
#el = ((554.8518676757812, 518.0960693359375), (265.05108642578125, 310.8172912597656), 90.39166259765625) # P5 Post v0
#el = ((470.3012390136719, 466.0242919921875), (194.33599853515625, 211.64601135253906), 94.243896484375) # P6 Pre v0
#el = ((483.29217529296875, 475.38385009765625), (182.44175720214844, 191.65750122070312), 124.80918884277344) # P7 Pre v0
#el = ((375.16070556640625, 363.8629455566406), (226.91879272460938, 233.74827575683594), 172.66668701171875) # P7 Post v0
el = ((403.751220703125, 408.5387878417969), (180.74961853027344, 193.22450256347656), 91.25155639648438) # P8 Post v0

track_fails = 0
prev_el = el
orientations = []
ellipses = []

frame_num = 0
#ellipses = pickle.load(open("ellipses.pkl", "rb"))
#centers = [el[0] for el in ellipses]
#o_centers = np.array(centers)
#for i in range(1, len(centers)-1):
#    avg = (o_centers[i]-1 + o_centers[i] + o_centers[i+1]) / 3
#    centers[i] = (int(avg[0]), int(avg[1]))

proc_noise_cov = 5e-2
meas_noise_cov = 5e-1
kalman_filter = KalmanFilter(proc_noise_cov, meas_noise_cov, el[0])
estimated = (0, 0)

while frame_good:
    """
    center = centers.pop(0)
    frame_good, image = read_scaled_img_with_border(capture, FRAME_BORDER_SIZE, FRAME_SCALE)
    cv2.circle(image, center, 3, (0,0,255), -1)
    cv2.imshow("Iris Tracker", image)
    #cv2.waitKey(int(1000./fps))
    cv2.waitKey()
    """

    track_success, el = update_ellipse(el, image)
    if el[1][0]*el[1][1] > 1.5*prev_el[1][0]*prev_el[1][1] or max(el[1])/min(el[1]) > 3:
        track_success = False

    if track_success:
        track_fails = 0

        estimated = kalman_filter.update(el[0])
        #el[0] = estimated
    else:
        el = list(prev_el)
        el[0] = kalman_filter.predict()
        track_fails += 1

    el = [(int(el[0][0]), int(el[0][1])), (int(el[1][0]), int(el[1][1])), el[2]]

    phi = np.pi * el[2] / 180.
    tau = np.arccos( min(*el[1]) / max(*el[1]))
    orientations.append((phi, tau))
    ellipses.append(el)

    if display:
        if track_success:
            cv2.ellipse(image, tuple(el), (0,255,0), 1)
        else:
            cv2.ellipse(image, tuple(el), (0,0,255), 2)

        cv2.circle(image, el[0], 3, (0,0,255), -1)
        cv2.circle(image, (int(estimated[0]), int(estimated[1])), 3, (0,255,0), -1)

        cv2.imshow("Iris Tracker", image)
        cv2.waitKey(int(1000./fps))
        #cv2.waitKey()
        #cv2.imwrite("out/%d.png" % frame_num, image)

    frame_good, image = read_scaled_img_with_border(capture, FRAME_BORDER_SIZE, FRAME_SCALE)

    if track_fails > 2:
        print("Tracking failed, pick a new ellipse.")
        el = pick_ellipse(image)
        if el is None: el = prev_el

    prev_el = el
    frame_num += 1
    if frame_num > 10*fps:
        break

print("Saving tracked ellipses...")
pickle.dump(ellipses, open("ellipses.pkl", "wb"))
