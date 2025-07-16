from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
import numpy as np
from datetime import datetime, timedelta
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.plotter import AnimatedPlotterly
import glob
import pandas as pd
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
def kalman_filter(s_s , s_c , chn_list ,nch):
    s_s_ = np.zeros(nch)
    s_c_ = np.zeros(nch)
    for i in range(len(chn_list.values)):
        s_s_[chn_list[i]] = s_s[i]
        s_c_[chn_list[i]] = s_c[i]
    # kalman_filter(p_s)

    measurement_model = LinearGaussian(
        ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
        mapping=(0, 2),  # Mapping measurement vector index to state index
        noise_covar=np.array([[5, 0],  # Covariance matrix for Gaussian PDF
                            [0, 5]])
        )

    start_time = datetime.now().replace(microsecond=0)
    timesteps = [start_time]
    measurements = []
    n_p = np.mean(s_s)
    for k in range(0, nch):
        timesteps.append(start_time+timedelta(seconds=k)) 
        if k%3 == 0:
            measurement = [[k, s_s_[k]]]
            # print(measurement)
            if s_s_[k]> 0 :
                if np.abs(measurement[0][1] - n_p) < 10 :
                    # print(n_p , measurement[0][1])
                    measurements.append(Detection(measurement,
                                                timestamp=timesteps[k],
                                                measurement_model=measurement_model))
                    n_p = measurement[0][1]

    # plotter = AnimatedPlotterly(timesteps, tail_length=0.3)  
    # plotter.plot_measurements(measurements, [0, 2])
    # plotter.fig

    q_x = 0.05
    q_y = 0.05
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                            ConstantVelocity(q_y)])
    from stonesoup.predictor.kalman import ExtendedKalmanPredictor
    predictor = ExtendedKalmanPredictor(transition_model)

    from stonesoup.updater.kalman import ExtendedKalmanUpdater
    updater = ExtendedKalmanUpdater(measurement_model)
    prior = GaussianState([[1002,], [1], [ np.float64(31.11)], [0]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
    track = Track()
    list = []
    for measurement in measurements:
        prediction = predictor.predict(prior, timestamp=measurement.timestamp)
        hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
        post = updater.update(hypothesis)
        track.append(post)
        prior = track[-1]
        list.append(hypothesis.measurement_prediction.state_vector.tolist())
    # plotter.plot_tracks(track, [0, 2], uncertainty=False)
    # plotter.fig
    return list