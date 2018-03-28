#include <iostream>
#include <vector>
#include <limits>
#include <cassert>
using std::vector;
using std::numeric_limits;

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

#include "ellipse.hpp"
#include "util.hpp"

static const int RANSAC_ITERS = 20000;

PointPair ellipse_foci(const RotatedRect el) {
    float F;
    float angle2rot = -el.angle;

    if(el.size.width > el.size.height) {
        F = sqrt(sqr(el.size.width/2.f) - sqr(el.size.height/2.f));
    } else {
        F = sqrt(sqr(el.size.height/2.f) - sqr(el.size.width/2.f));
        angle2rot += 90;
    }

    const auto F1 = point_on_circle(el.center, F, deg2rad(angle2rot));
    const auto F2 = 2*el.center - F1;
    return {F1, F2};
}

vector<PointPair> get_tracker_lines(const RotatedRect el,
                                    const int len, const int npts,
                                    const int start, const int stop) {

    const int half_len = (len > 0) ? (len / 2) :
        ((el.size.width + el.size.height) / 20);

    vector<PointPair> lines;
    for(auto pt : ellipse_points(el, npts, start, stop)) {
        auto norm = Point2f(ellipse_normal(el, pt));
        lines.push_back({pt - half_len*norm, pt + half_len*norm});
    }
    return lines;
}

pair<RotatedRect, vector<int>> fit_ellipse_ransac(const vector<Point2f> pts, const float inlier_thresh) {
    assert((pts.size() >= 6) && "RANSAC ellipse fitting requires at least 6 points.");
    vector<int> best_inliers;

    for(int it=0; it<RANSAC_ITERS; ++it) {
        const auto el = fitEllipse(random_sample(pts, 5));

        vector<int> inlier_ids;
        for(size_t i=0; i<pts.size(); ++i) {
            if (ellipse_fit_error(el, pts[i]) < inlier_thresh) {
                inlier_ids.push_back(i);
            }
        }

        if (inlier_ids.size() > best_inliers.size()) {
            best_inliers = inlier_ids;
        }
    }

    return {fitEllipse(take(pts, best_inliers)), best_inliers};
}

vector<Point2f> compute_EPIC_points(const vector<LineIterator> tracker_lines, const Mat image, const Mat grad) {
    vector<Point2f> proposed_ellipse_pts;

    const int mean_winsize = tracker_lines[0].count / 5.f;
    Mat edges;
    Canny(image, edges, 50, 100);
    imshow("edges", edges);

    int lic=0;
    for(auto li : tracker_lines) {
        vector<size_t> local_maxima_indices;
        float max_grad_mag = numeric_limits<float>::min();

        Point2f p0, p1, p2, p3, p4;
        float gp0=0, gp1=0, gp2=0, gp3=0, gp4=0;
        vector<Point2f> line_pts(li.count);
        vector<float> intensity_sums(li.count, 0);
        vector<float> inner_minimums(li.count, 0);

        // Compute local gradient maxima along each tracker line and
        // central intensity differences for each local maximum.
        for(int i=0; i<li.count; ++i, ++li) {
            p0 = p1; p1 = p2; p2 = p3; p3 = p4; p4 = li.pos();
            gp0 = gp1; gp1 = gp2; gp2 = gp3; gp3 = gp4; gp4 = grad.at<float>(p4);
            line_pts[i] = p4;

            if (i > mean_winsize+2 && i < (li.count-mean_winsize+1) &&
                (gp1+gp2+gp3) > (gp0+gp1+gp2) &&
                (gp1+gp2+gp3) > (gp2+gp3+gp4)) {

                local_maxima_indices.push_back(i-2);
                max_grad_mag = max(max_grad_mag, gp2);
            }

            if(edges.at<uint8_t>(p4) && i >= 2) {
                local_maxima_indices.push_back(i-2);
            }

            const float intensity = image.at<uint8_t>(p4) / 255.f;
            intensity_sums[i] += intensity;
            inner_minimums[i] = intensity;
            if(i > 0) {
                inner_minimums[i] = min(intensity, inner_minimums[i-1]);
                intensity_sums[i] += intensity_sums[i-1];
            }
        }

        if(local_maxima_indices.size() == 0) { continue; }

        // Select the local gradient maximum with the highest likelihood
        // heuristic value on each tracker line.
        size_t best_index = local_maxima_indices[0];
        const float w0 = 0.5f, w1 = 1.0f, w2 = 0.0f;
        float max_likelihood = numeric_limits<float>::min();
        float best_intensity_ratio = numeric_limits<float>::min();
        float second_best_ratio = 0.f;
        for(auto ind : local_maxima_indices) {
            const float xm = grad.at<float>(line_pts[ind]);

            const float inner_mean = (intensity_sums[ind-1] - intensity_sums[ind-mean_winsize-1]) / (float)mean_winsize;
            const float outer_mean = (intensity_sums[ind+mean_winsize+1] - intensity_sums[ind]) / (float)mean_winsize;

            const float C0 = (xm == max_grad_mag) ? 1 : 0;
            const float C1 = (abs(inner_mean) > 1e-8) ? (outer_mean / inner_mean) : 0;
            const float C2 = 1.f - inner_mean;

            const float edge_likelihood = (w0*C0 + w1*C1 + w2*C2) / (w0+w1+w2);

            if (edge_likelihood > max_likelihood) {
                second_best_ratio = edge_likelihood / max_likelihood;
                max_likelihood = edge_likelihood;
                best_index = ind;
                best_intensity_ratio = C1;
            }
        }

        const Point2f& best_pt = line_pts[best_index];

        if(
           best_intensity_ratio > 1.5f// &&
           //inner_minimums[best_index] > 0.1f &&
           //inner_minimums[best_index] < 0.5f
        ) {
            proposed_ellipse_pts.push_back(best_pt);
        }

        ++lic;
    }

    if(proposed_ellipse_pts.size() > 0) {
        const auto crit = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 1e-3);
        cornerSubPix(image, proposed_ellipse_pts, {5,5}, {-1, -1}, crit);
    }

    return proposed_ellipse_pts;
}
