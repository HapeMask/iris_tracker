#include <Python.h>
#include <numpy/arrayobject.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
using namespace cv;

#include "tracking.hpp"
#include "util.hpp"

static PyObject* py_update_ellipse(PyObject* self, PyObject* args) {
    float cx, cy, height, width, angle;
    int allow_approx, line_len, n_lines;
    PyArrayObject* np_image;

    if(!PyArg_ParseTuple(args, "fffffO!iii",
                &cx, &cy, &width, &height, &angle,
                &PyArray_Type, &np_image,
                &allow_approx, &line_len, &n_lines)){
        return NULL;
    }

    assert(PyArray_NDIM(np_image) == 3); // HxWxBGR
    assert(PyArray_SHAPE(np_image)[2] == 3); // Color Only
    assert(PyArray_TYPE(np_image) == NPY_UINT8); // 24-bit BGR

    Mat image(PyArray_SHAPE(np_image)[0], PyArray_SHAPE(np_image)[1], CV_8UC3, (unsigned char*)PyArray_DATA(np_image));
    RotatedRect el{{cx, cy}, {width, height}, angle};
    long success = update_ellipse(el, image, allow_approx, line_len, n_lines);

    PyObject* el_tuple = PyTuple_New(3);
    PyObject* center_tuple = PyTuple_New(2);
    PyObject* size_tuple = PyTuple_New(2);

    PyTuple_SET_ITEM(center_tuple, 0, PyFloat_FromDouble(el.center.x));
    PyTuple_SET_ITEM(center_tuple, 1, PyFloat_FromDouble(el.center.y));
    PyTuple_SET_ITEM(size_tuple, 0, PyFloat_FromDouble(el.size.width));
    PyTuple_SET_ITEM(size_tuple, 1, PyFloat_FromDouble(el.size.height));

    PyTuple_SET_ITEM(el_tuple, 0, center_tuple);
    PyTuple_SET_ITEM(el_tuple, 1, size_tuple);
    PyTuple_SET_ITEM(el_tuple, 2, PyFloat_FromDouble(el.angle));

    PyObject* ret = PyTuple_New(2);
    PyTuple_SET_ITEM(ret, 0, PyBool_FromLong(success));
    PyTuple_SET_ITEM(ret, 1, el_tuple);
    return ret;
}

static PyObject* py_make_kalman(PyObject* self, PyObject* args) {
    float process_noise_cov = 0.f;
    float measurement_noise_cov = 0.f;
    float i_x = 0.f, i_y = 0.f;

    if(!PyArg_ParseTuple(args, "ffff", &process_noise_cov, &measurement_noise_cov, &i_x, &i_y)){
        return NULL;
    }

    const Point2f init_pos = {i_x, i_y};
    auto kf = init_kalman_2d(process_noise_cov, measurement_noise_cov, init_pos);
    KalmanFilter* kfp = new KalmanFilter(kf);
    return PyLong_FromVoidPtr((void*)kfp);
}

static PyObject* py_update_kalman(PyObject* self, PyObject* args) {
    PyObject* pkfp;
    float p_x = 0.f, p_y = 0.f;

    if(!PyArg_ParseTuple(args, "ffO", &p_x, &p_y, &pkfp)){
        return NULL;
    }

    const Point2f new_pos = {p_x, p_y};
    KalmanFilter* kfp = (KalmanFilter*)PyLong_AsVoidPtr(pkfp);

    kfp->predict();
    Mat estimated = kfp->correct(Mat(new_pos));

    const Point2f estimated_pt = Point2f(estimated(Range(0,2),Range::all()));
    PyObject* ret = PyTuple_New(2);
    PyTuple_SET_ITEM(ret, 0, PyFloat_FromDouble((double)estimated_pt.x));
    PyTuple_SET_ITEM(ret, 1, PyFloat_FromDouble((double)estimated_pt.y));

    return ret;
}

static PyObject* py_kalman_predict(PyObject* self, PyObject* args) {
    PyObject* pkfp;

    if(!PyArg_ParseTuple(args, "O", &pkfp)){
        return NULL;
    }

    KalmanFilter* kfp = (KalmanFilter*)PyLong_AsVoidPtr(pkfp);

    Mat prediction = kfp->predict();
    const Point2f pt = Point2f(prediction(Range(0,2),Range::all()));
    PyObject* ret = PyTuple_New(2);
    PyTuple_SET_ITEM(ret, 0, PyFloat_FromDouble((double)pt.x));
    PyTuple_SET_ITEM(ret, 1, PyFloat_FromDouble((double)pt.y));

    return ret;
}

static PyObject* py_delete_kalman(PyObject* self, PyObject* args) {
    PyObject* pkfp;

    if(!PyArg_ParseTuple(args, "O", &pkfp)){
        return NULL;
    }

    KalmanFilter* kfp = (KalmanFilter*)PyLong_AsVoidPtr(pkfp);
    delete kfp;
    Py_RETURN_NONE;
}

static PyMethodDef iris_trackMethods[] = {
    {"make_kalman", py_make_kalman, METH_VARARGS, "Create and initialize a Kalman filter."},
    {"update_kalman", py_update_kalman, METH_VARARGS, "Update a Kalman filter."},
    {"kalman_predict", py_kalman_predict, METH_VARARGS, "Predict a point with a Kalman filter."},
    {"delete_kalman", py_delete_kalman, METH_VARARGS, "Clean up a Kalman filter."},
    {"update_ellipse", py_update_ellipse, METH_VARARGS, "Track an existing ellipse in a new image."},
    {NULL, NULL, 0, NULL}
};

extern "C" {
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef iris_track_moduledef =
{
    PyModuleDef_HEAD_INIT,
    "iris_track",
    "Iris Tracker",
    -1,     /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
    iris_trackMethods
};

PyMODINIT_FUNC PyInit_iris_track(void)
#else
PyMODINIT_FUNC initiris_track(void)
#endif
{
    PyObject* m;
#if PY_MAJOR_VERSION == 3
    m = PyModule_Create(&iris_track_moduledef);
#else
    m = Py_InitModule3("iris_track", iris_trackMethods, "Iris Tracker");
#endif

    import_array();

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}
}
