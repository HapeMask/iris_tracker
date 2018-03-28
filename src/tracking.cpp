#include <iostream>
#include <vector>
using std::vector;

#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

#include "tracking.hpp"
#include "util.hpp"
#include "ellipse.hpp"

bool update_ellipse(RotatedRect& el, const Mat& image, const bool allow_approx, int line_len, int n_lines) {
    Size unused; Point2f roi_pos; Point _roi;
    Mat tracked_region(image, el.boundingRect());

    tracked_region.adjustROI(50,50,50,50);
    tracked_region.locateROI(unused, _roi);
    roi_pos = {(float)_roi.x, (float)_roi.y};

    // Get tracker lines on the current ellipse and offset the positions so
    // that they are relative to the tracked ROI.
    vector<LineIterator> line_its;
    for (auto& ln : get_tracker_lines(el, line_len, n_lines)) {
        line_its.push_back(LineIterator{image,
                ln.first - roi_pos,
                ln.second - roi_pos});
    }

    // Convert to greyscale and compute the gradient magnitude in the ROI.
    cvtColor(tracked_region, tracked_region, COLOR_BGR2GRAY);
    medianBlur(tracked_region, tracked_region, 9);
    equalizeHist(tracked_region, tracked_region);
    imshow("tr", tracked_region);
    const Mat grad = normalize_image(gradient_mag(tracked_region));

    // Find points to track using EPIC, then fit an ellipse to them w/RANSAC.
    bool ransac_success = false;
    auto proposed_ellipse_pts = compute_EPIC_points(line_its, tracked_region, grad);

    if(proposed_ellipse_pts.size() > n_lines/4) {
        const auto new_el = fit_ellipse_ransac(proposed_ellipse_pts, 2);
        const float inlier_fraction = new_el.second.size() / (float)proposed_ellipse_pts.size();

        for(size_t i=0; i<proposed_ellipse_pts.size(); ++i) {
            if(in(i, new_el.second)) {
                circle(image, Point2f(roi_pos)+proposed_ellipse_pts[i], 1, {255,255,0}, -1);
            } else {
                circle(image, Point2f(roi_pos)+proposed_ellipse_pts[i], 1, {255,0,0}, -1);
            }
        }

        if(inlier_fraction > MIN_INLIER_FRACTION) {
            ransac_success = true;
            el = new_el.first;
        }
    }

    // If EPIC+RANSAC didn't find enough good points (usually because the
    // eye was moving too fast), try Hough circle detection to get a rough
    // estimate instead.
    if(!ransac_success && allow_approx) {
        const float radius = (el.size.width+el.size.height)/4.f;
        try {
            el = best_hough_circle(tracked_region, radius);
        } catch(...) {
            return false;
        }
    } else if(!ransac_success) {
        return false;
    }

    el.center += Point2f(roi_pos);
    return true;
}
