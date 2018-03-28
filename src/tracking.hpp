#pragma once
#include "opencv2/core/core.hpp"
using namespace cv;

static const float MIN_INLIER_FRACTION = 0.2;

bool update_ellipse(RotatedRect& el, const Mat& image, const bool allow_approx,
        int line_len, int n_lines);
