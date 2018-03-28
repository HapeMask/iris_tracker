#pragma once

#include <vector>
#include <ostream>
#include <numeric>
#include <algorithm>
#include <random>
using std::vector;
using std::ostream;
using std::pair;

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core.hpp"
using namespace cv;

#define sqr(x) ((x)*(x))

// Shared PRNG used for random sampling.
static std::mt19937 rng(0);

inline float rad2deg(const float radians) { return 180.f * radians / M_PI; }
inline float deg2rad(const float degrees) { return M_PI * degrees / 180.f; }

/**
 * Returns the point on a circle with center 'c', radius 'r', at a given angle.
 */
inline Point2f point_on_circle(const Point2f c, const float r, const float angle) {
    return c + Point2f(cos(angle), -sin(angle))*r;
}

/**
 * Compute the squared gradient magnitude of an image.
 */
inline Mat gradient_mag(const Mat img) {
    Mat gx, gy;
    Scharr(img, gx, CV_32F, 1, 0);
    Scharr(img, gy, CV_32F, 0, 1);
    return gx.mul(gx) + gy.mul(gy);
}

/**
 * Predicate for whether a list 'c' contains value 'v'. 
 */
template <template<typename...> class Container, typename Value, typename... CTypes>
inline bool in(const Value& v, const Container<CTypes...>& c) {
    return (find(c.begin(), c.end(), v) != c.end());
}

/*
 * Selects items from container 'v' with indices in 'int'. 
 */
template <typename Container>
inline Container take(const Container& v, const vector<int> ind) {
    Container out;
    for(const auto i : ind) { out.push_back(v[i]); }
    return out;
}

/**
 * Randomly sample 'size' elements from container 'in' without replacement.
 */
template <typename Container>
inline Container random_sample(const Container& in, const int size) {
    vector<int> indices(in.size());
    std::iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), rng);

    return take(in, {indices.begin(), indices.begin()+size});
}

/*
 * Scales an image so that its color values lie within the range [0,1].
 */
inline Mat normalize_image(const Mat img) {
    Mat out;
    img.convertTo(out, CV_32F);

    double minval, maxval;
    minMaxLoc(out.reshape(1), &minval, &maxval);
    return (out - minval) / (maxval - minval);
}

inline Mat read_scaled_img_with_border(VideoCapture& cap, const int border_size, const float scale) {
    Mat img, out;
    cap >> img;
    resize(img, img, {0,0}, scale, scale);
    copyMakeBorder(img, out, border_size, border_size, border_size, border_size, BORDER_CONSTANT);
    return out;
}

/**
 * Allow a user to pick an ellipse from an image by picking and moving 5
 * points.
 */
struct PickerData {
    vector<Point2f> picked_points;
    RotatedRect picked_ellipse;
    int nearest_id;
    bool drag_start = false, dragging = false, dirty = false;
    const Mat* iptr;
};

void ellipse_picker_callback(int event, int x, int y, int, void* v_data);
RotatedRect pick_ellipse(const Mat image);

/**
 * Use OpenCV's Hough circle detector to find the best circle with a given
 * radius (+/- radius*range) in an image.
 */
inline RotatedRect best_hough_circle(const Mat image, float const radius, const float range=0.1) {
    vector<Vec3f> circles;
    HoughCircles(image, circles,
            CV_HOUGH_GRADIENT,
            4, 64, 100, 100,
            (1.f-range)*radius,
            (1.f+range)*radius);

    if(circles.size() < 1) { throw("Hough circle detection failed."); }
    return {{circles[0][0],circles[0][1]}, {2.f*circles[0][2], 2.f*circles[0][2]}, 0};
}

/**
 * Construct a Kalman filter to track 2D points.
 */
inline KalmanFilter init_kalman_2d(const float process_noise_cov,
                                   const float measurement_noise_cov,
                                   const Point2f init_pos={0.f,0.f})
{

    KalmanFilter kf(4, 2);
    setIdentity(kf.measurementMatrix);
    setIdentity(kf.processNoiseCov, process_noise_cov);
    setIdentity(kf.measurementNoiseCov, measurement_noise_cov);
    kf.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0,  // x + dx
                                                0, 1, 0, 1,  // y + dy
                                                0, 0, 1, 0,  // dx
                                                0, 0, 0, 1); // dy
    kf.statePost.at<float>(0) = init_pos.x;
    kf.statePost.at<float>(1) = init_pos.y;
    kf.statePost(Range(2,4), Range::all()) = Scalar(0);

    // We assume that the user's initial input is almost exact.
    setIdentity(kf.errorCovPost, 1e-5);

    return kf;
}

template <typename T1, typename T2>
inline ostream& operator<<(ostream& out,  const pair<T1, T2>& p) {
    out << "pair<" << p.first << ", " << p.second << ">";
    return out;
}

template <typename T>
inline ostream& operator<<(ostream& out, const vector<T>& v) {
    out << "[ ";
    for(const auto& e : v) {
        out << e << " ";
    }
    out << "]";
    return out;
}
