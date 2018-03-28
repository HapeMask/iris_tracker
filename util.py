import numpy as np
from scipy.misc import imresize
import cv2

PICKER_WINDOW_NAME = "Iris Picker"

def read_scaled_img_with_border(capture, border_size, scale):
    frame_good, img = capture.read()
    if frame_good:
        img = imresize(img, (int(scale*img.shape[0]), int(scale*img.shape[1])))

        padded = np.zeros((img.shape[0] + 2*border_size, img.shape[1] + 2*border_size, 3), img.dtype)
        padded[border_size:-border_size, border_size:-border_size] = img
    else:
        padded = None

    return frame_good, padded

def ellipse_picker_callback(event, x, y, ignored, data):
    image = data[6]
    mouse_pos = np.array((x, y))

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        data[4] = False
        data[3] = True

        if len(data[0]) < 5:
            data[0].append(mouse_pos)
            if len(data[0]) == 5:
                data[1] = cv2.fitEllipse(np.array(data[0]))
            data[5] = True
        else:
            # Once the user has picked 5 points, switch to move mode.
            # Clicking and data[4] selects the nearest point and moves it to
            # the cursor.

            # Find the nearest picked point to the clicked position.
            min_dist = np.linalg.norm(mouse_pos - data[0][0])
            data[2] = 0
            for i in range(1,5):
                if np.linalg.norm(mouse_pos - data[0][i]) < min_dist:
                    min_dist = np.linalg.norm(mouse_pos - data[0][i])
                    data[2] = i

            if np.linalg.norm(mouse_pos - data[1][0]) < min_dist:
                data[2] = -1
    elif event == cv2.EVENT_LBUTTONUP:
        data[4] = False
        data[3] = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if data[3]:
            data[4] = True
            data[3] = False
        elif data[4] and len(data[0]) == 5:
            # If we are in move mode, update the position of the picked point
            # to follow the cursor.
            if data[2] >= 0:
                data[0][data[2]] = mouse_pos
                data[1] = cv2.fitEllipse(np.array(data[0]))
            else:
                for pt in data[0]:
                    pt += mouse_pos - data[1][0]
                data[1] = cv2.fitEllipse(np.array(data[0]))
            data[5] = True
    elif event == cv2.EVENT_RBUTTONUP:
        data[5] = True

    # This callback is called on mouse movement and presses, we only want to
    # redraw if something changed.
    if data[5]:
        pick_image = image.copy()
        for pt in data[0]:
            cv2.circle(pick_image, tuple(pt), 3, (0,255,0), -1)

        if len(data[0]) >= 5:
            cv2.ellipse(pick_image, data[1], (255,255,255))
            el_center = (int(data[1][0][0]), int(data[1][0][1]))
            cv2.circle(pick_image, el_center, 3, (0,0,255), -1)

        cv2.imshow(PICKER_WINDOW_NAME, pick_image)
        data[5] = False

def pick_ellipse(image):
    data = [[], [], 0, False, False, False, image]

    cv2.namedWindow(PICKER_WINDOW_NAME)
    cv2.setMouseCallback(PICKER_WINDOW_NAME, ellipse_picker_callback, data)
    cv2.startWindowThread()
    cv2.imshow(PICKER_WINDOW_NAME, image)

    key = None
    while key != '\n' and key != '\r' and key != 'q':
        key = cv2.waitKey(33)
        key = chr(key & 255) if key >= 0 else None

    cv2.destroyAllWindows()

    if key == 'q':
        return None
    else:
        return data[1]
