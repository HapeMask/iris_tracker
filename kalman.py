from iris_track import make_kalman, update_kalman, kalman_predict, delete_kalman

class KalmanFilter:
    def __init__(self, process_noise_cov, meas_noise_cov, init_pt=(0,0)):
        ix, iy = init_pt
        ix = float(ix); iy = float(iy)
        self.kfp = make_kalman(process_noise_cov, meas_noise_cov, ix, iy)

    def update(self, pt):
        px, py = pt
        px = float(px); py = float(py)
        return update_kalman(px, py, self.kfp)

    def predict(self):
        return kalman_predict(self.kfp)

    def __del__(self):
        if hasattr(self, "kfp"):
            delete_kalman(self.kfp)
