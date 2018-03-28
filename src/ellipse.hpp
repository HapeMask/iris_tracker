#pragma once
#include <vector>
#include <cmath>

#include "opencv2/core/core.hpp"
using namespace cv;

#include "util.hpp"

typedef pair<Point2f, Point2f> PointPair;

/*
 * Compute the two focal points for ellipse 'el'.
 */
PointPair ellipse_foci(const RotatedRect el);

/**
 * Returns a point on ellipse 'el' in parametric form at position 't'. Note
 * that 't' is not an angle.
 */
inline Point2f point_on_ellipse(const RotatedRect el, const float t) {
    const float sin_t = sin(t), cos_t = cos(t), phi = deg2rad(el.angle);

    const float x = el.center.x + (el.size.width / 2.f) * cos_t * cos(phi) -
        (el.size.height / 2.f) * sin_t * sin(phi);
    const float y = el.center.y + (el.size.width / 2.f) * cos_t * sin(phi) +
        (el.size.height / 2.f) * sin_t * cos(phi);
    return {x, y};
}

/**
 * Returns 'npts' evenly spaced points along the perimeter of ellipse 'el' in the interval
 * t=['start', 'stop'].
 */
inline vector<Point2f> ellipse_points(
        const RotatedRect el,
        const int npts,
        const float start=0, const float stop=2*M_PI) {

    vector<Point2f> out;
    for(int i = 0; i < npts; ++i) {
        const float t = (float)i / npts;
        out.push_back(point_on_ellipse(el, start + (stop-start) * t));
    }

    return out;
}

/**
 * Computes the normal to ellipse 'el' through point 'p'.
 */
inline Vec2f ellipse_normal(const RotatedRect el, const Point2f p) {
    const auto foci = ellipse_foci(el);
    const auto norm = normalize(Vec2f(p - foci.first)) +
                      normalize(Vec2f(p - foci.second));
    return normalize(norm);
}

/**
 * Constructs 'npts' tracker lines normal to ellipse 'el' evenly spaced on the
 * interval t=['start', 'stop'].
 */
vector<PointPair> get_tracker_lines(const RotatedRect el,
        const int len = 0, int const npts = 20,
        const int start = 0, const int stop = 2*M_PI);

inline void draw_tracker(Mat& img, const RotatedRect el, const vector<PointPair> lines) {
    ellipse(img, el, {255,255,255}, 2);
    circle(img, el.center, 3, {0,0,255}, -1);

    for(auto ln : lines) {
        line(img, ln.first, ln.second, {0,255,0}, 2);
    }
}

/**
 * Returns a pair <f, |g|> where f is the ellipse equation for ellipse 'el'
 * evaluated at point 'pt' and |g| is the gradient magnitude of f at 'pt'.
 */
inline pair<float, float> ellipse_equation_and_grad(const RotatedRect el, const Point2f pt) {
    const float sin_theta = sin(deg2rad(el.angle));
    const float cos_theta = cos(deg2rad(el.angle));

    const float a2 = sqr(min(el.size.width, el.size.height) / 2.f);
    const float b2 = sqr(max(el.size.width, el.size.height) / 2.f);
    const float A = b2*sqr(cos_theta) + a2*sqr(sin_theta);
    const float B = b2*sqr(sin_theta) + a2*sqr(cos_theta);
    const float C = 2.f * sin_theta * cos_theta * (b2 - a2);
    const float D = -2 * el.center.x * A - el.center.y * C;
    const float E = -2 * el.center.y * B - el.center.x * C;
    const float F = A * sqr(el.center.x) +
                    B * sqr(el.center.y) +
                    C * el.center.x * el.center.y -
                    a2 * b2;

    return {A*sqr(pt.x) + B*sqr(pt.y) + C*pt.x*pt.y + D * pt.x + E * pt.y + F,
            sqrt(sqr(2.f*A*pt.x + C*pt.y + D) + sqr(E + C*pt.x + 2.f*B*pt.y))
            };
}

/**
 * Computes the gradient-weighted algebraic distance between point 'pt' and
 * ellipse 'el'.
 */
inline float ellipse_fit_error(const RotatedRect el, const Point2f pt) {
    const auto q = ellipse_equation_and_grad(el, pt);
    return abs(q.first / q.second);
}

/**
 * Finds the most likely point along each line in 'tracker_lines' to lie on the
 * eye's limbus boundary, using (something like) EPIC.
 */
vector<Point2f> compute_EPIC_points(const vector<LineIterator> tracker_lines, const Mat image, const Mat grad);

/** Fits the best ellipse to the point set 'pts'. Returns a pair <R, I> where R
 * is the ellipse and I is a list of indices corresponding to inlier points.
 * Inliers are points whose ellipse_fit_error() is less than 'inlier_thresh'.
 */
pair<RotatedRect, vector<int>> fit_ellipse_ransac(const vector<Point2f> pts, const float inlier_thresh = 2);
