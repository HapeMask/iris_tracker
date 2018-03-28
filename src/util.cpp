#include <vector>
using std::vector;

#include "opencv2/highgui/highgui.hpp"
using namespace cv;

#include "util.hpp"

#include <iostream>
using namespace std;
void ellipse_picker_callback(int event, int x, int y, int, void* v_data) {
    PickerData* data = static_cast<PickerData*>(v_data);
    vector<Point2f>& picked_points = data->picked_points;
    RotatedRect& picked_ellipse = data->picked_ellipse;
    int& nearest_id = data->nearest_id;
    bool& drag_start = data->drag_start;
    bool& dragging = data->dragging;
    bool& dirty = data->dirty;
    const Mat*& iptr = data->iptr;

    const float xf = x, yf = y;

    Point2f cur_pt{xf, yf};

    if (event == EVENT_LBUTTONDOWN || event == EVENT_RBUTTONDOWN) {
        dragging = false;
        drag_start = true;

        if(picked_points.size() < 5) {
            picked_points.push_back({xf, yf});
            if (picked_points.size() == 5) {
                picked_ellipse = fitEllipse(picked_points);
            }
            dirty = true;
        } else {
            // Once the user has picked 5 points, switch to move mode.
            // Clicking and dragging selects the nearest point and moves it to
            // the cursor.

            // Find the nearest picked point to the clicked position.
            float min_dist = norm(cur_pt - picked_points[0]);
            nearest_id = 0;
            for(int i=1; i<5; ++i) {
                if(norm(cur_pt - picked_points[i]) < min_dist) {
                    min_dist = norm(cur_pt - picked_points[i]);
                    nearest_id = i;
                }
            }

            if (norm(cur_pt - picked_ellipse.center) < min_dist) {
                nearest_id = -1;
            }
        }
    } else if (event == EVENT_LBUTTONUP) {
        dragging = false;
        drag_start = false;
    } else if (event == EVENT_MOUSEMOVE) {
        if(drag_start) {
            dragging = true;
            drag_start = false;
        } else if (dragging && picked_points.size() == 5) {
            // If we are in move mode, update the position of the picked point
            // to follow the cursor.
            if(nearest_id >= 0) {
                picked_points[nearest_id] = cur_pt;
                picked_ellipse = fitEllipse(picked_points);
            } else {
                for(auto& pt : picked_points) { pt += (cur_pt - picked_ellipse.center); }
                picked_ellipse = fitEllipse(picked_points);
            }
            dirty = true;
        }
    } else if (event == EVENT_RBUTTONUP) {
        dirty = true;
    }

    // This callback is called on mouse movement and presses, we only want to
    // redraw if something changed.
    if(dirty) {
        Mat pick_image = ((Mat*)iptr)->clone();
        for(const auto& pt : picked_points) {
            circle(pick_image, pt, 3, {0,255,0}, -1);
        }

        if (picked_points.size() >= 5) {
            ellipse(pick_image, picked_ellipse, {255,255,255});
            circle(pick_image, picked_ellipse.center, 3, {0,0,255}, -1);
        }

        imshow("Iris Picker", pick_image);
        dirty = false;
        pick_image.release();
    }
}

RotatedRect pick_ellipse(const Mat image) {
    PickerData data;
    data.iptr = &image;

    namedWindow("Iris Picker");
    setMouseCallback("Iris Picker", ellipse_picker_callback, (void*)&data);
    imshow("Iris Picker", image);

    int key = 0;
    while(key != '\n' && key != '\r') {
        key = waitKey(66);
        if(key == 'q') {
            return RotatedRect();
        }
    }

    destroyWindow("Iris Picker");
    return data.picked_ellipse;
}
