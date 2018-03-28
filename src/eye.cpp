#include <iostream>
#include <vector>
using std::cerr;
using std::endl;
using std::vector;

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
using namespace cv;

#include "util.hpp"
#include "tracking.hpp"
#include "ellipse.hpp"

static const int FRAME_BORDER_SIZE = 100;
static const float FRAME_SCALE = 0.5;

static const float KALMAN_PROCESS_NOISE = 1e-2;
static const float KALMAN_MEAS_NOISE = 1e-1;

int main(int argc, char* args[]) {
    bool write_output = (argc == 3);
    if(argc < 2) {
        cerr << "USAGE: eye VIDEO_FILE [OUT_FILE]" << endl;
        return 1;
    }

    auto in_vid = VideoCapture(args[1]);
    if(!in_vid.isOpened()) {
        cerr << "Failed to open input video '" << args[1] << "' for reading." << endl;
        return 1;
    }
    const auto fps = in_vid.get(CV_CAP_PROP_FPS);

    Mat image = read_scaled_img_with_border(in_vid, FRAME_BORDER_SIZE, FRAME_SCALE);

    cerr << "Click 5 points to form an ellipse." << endl;
    cerr << "Points may be adjusted after picking 5." << endl;
    cerr << "Press enter when done, 'q' to quit." << endl;

    auto el = pick_ellipse(image);
    // DSC_0103
    //RotatedRect el{{712.912, 330.851}, {249.825, 264.771}, 10.1404};

    if (el.size.width == 0 && el.size.height == 0) {
        cerr << "Aborting iris tracker." << endl;
        return 1;
    }

    VideoWriter out_vid;
    const Size out_sz = boundingRect(ellipse_points(el, 10)).size()+Size{40,40};
    if(write_output) {
        out_vid.open(args[2], CV_FOURCC('X', '2', '6', '4'), fps, out_sz, true);
        if(!out_vid.isOpened()) {
            cerr << "Failed to open output video '" << args[2] << "' for writing." << endl;
            return 1;
        }
    }

    auto kalman_filter = init_kalman_2d(KALMAN_PROCESS_NOISE, KALMAN_MEAS_NOISE, el.center);
    int track_fails = 0;
    auto prev_el = el;

    // Matrices for Kalman filter predictions.
    Mat predicted;
    Mat estimated;
    Point2f estimated_pt;
    for(int frame=1; frame < (int)in_vid.get(CV_CAP_PROP_FRAME_COUNT); ++frame) {
        bool track_success = update_ellipse(el, image, true, 20, 40);

        if(track_success) {
            track_fails = 0;

            // Update the Kalman filter using the new ellipse.
            predicted = kalman_filter.predict();
            estimated = kalman_filter.correct(Mat(el.center));
            estimated_pt = Point2f(estimated(Range(0,2),Range::all()));
        } else {
            el = prev_el;
            el.center = Point2f(kalman_filter.predict()(Range(0,2),Range::all()));
            ++track_fails;
        }

        if(track_success) {
            ellipse(image, el, {0,255,0}, 1);
        } else {
            ellipse(image, el, {0,0,255}, 2);
        }

        circle(image, el.center, 3, {0,0,255}, -1);
        circle(image, estimated_pt, 3, {0,255,0}, -1);

        imshow("Iris Tracker", image);
        waitKey(1000.f/fps);

        if(write_output) {
            out_vid << image(Rect{estimated_pt-0.5f*Point2f(out_sz), out_sz});
        }

        image = read_scaled_img_with_border(in_vid, FRAME_BORDER_SIZE, FRAME_SCALE);
        prev_el = el;
    }

    return 0;
}
