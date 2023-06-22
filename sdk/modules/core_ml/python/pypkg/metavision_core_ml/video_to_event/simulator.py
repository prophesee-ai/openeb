# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
EventSimulator: Load a .mp4 video and start streaming events
"""

import numpy as np
from metavision_core_ml.video_to_event.single_image_make_events_cpu import EventCPU


def eps_log(x, eps=1e-5):
    """
    Takes Log of image

    Args:
        x: uint8 gray frame
    """
    if x.dtype is not np.float32:
        x = x.astype(np.float32)
    return np.log(eps + x / 255.0)


class EventSimulator(object):
    """Event Simulator

    Implementation is based on the following publications:

    - Video to Events: Recycling Video Datasets for Event Cameras: Daniel Gehrig et al.
    - V2E: From video frames to realistic DVS event camera streams: Tobi Delbruck et al.

    This object allows to accumulate events by feeding it with images and (increasing) timestamps.
    The events are returned of type EventCD (see definition in event_io/dat_tools or metavision_sdk_base)

    Args:
        Cp (float): mean for ON threshold
        Cn (float): mean for OFF threshold
        refractory_period (float): min time between 2 events / pixel
        sigma_threshold (float): standard deviation for threshold array
        cutoff_hz (float): cutoff frequency for photodiode latency simulation
        leak_rate_hz (float): frequency of reference value leakage
        shot_noise_rate_hz (float): frequency for shot noise events
    """

    def __init__(
            self, height, width, Cp, Cn, refractory_period, sigma_threshold=0.0, cutoff_hz=0, leak_rate_hz=0,
            shot_noise_rate_hz=0, verbose=False):
        self.Cp = Cp
        self.Cn = Cn
        self.refractory_period = refractory_period
        self.ref_values = None
        self.last_event_timestamp = None
        self.events = []
        self.last_img = None
        self.last_img_ts = 0.
        self.shot_noise_rate_hz = shot_noise_rate_hz
        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.sigma_threshold = sigma_threshold
        self.height = height
        self.width = width
        self.ref_values = np.zeros((height, width), dtype=np.float64)
        self.last_img = np.zeros_like(self.ref_values)
        self.last_event_timestamp = np.zeros((height, width), dtype=np.int32)
        self.first_pass = True

        # Threshold Mismatches
        self.Cps = self.sigma_threshold * np.random.randn(height, width).astype(np.float32) + self.Cp
        if verbose and ((self.Cps < 0.01).any() or (self.Cps > 1).any()):
            print(
                f"WARNING: EventSimulator: Some Cp are out of range: min: {self.Cps.min()}  max: {self.Cps.max()}. Clipping to range [0.01;1]")
        self.Cps = np.clip(self.Cps, 0.01, 1)
        self.Cns = self.sigma_threshold * np.random.randn(height, width).astype(np.float32) + self.Cn
        if verbose and ((self.Cns < 0.01).any() or (self.Cns > 1).any()):
            print(
                f"WARNING: EventSimulator: Some Cn are out of range: min: {self.Cns.min()}  max: {self.Cns.max()}. Clipping to range [0.01;1]")
        self.Cns = np.clip(self.Cns, 0.01, 1)

        self.lpLogFrame0 = None
        self.lpLogFrame1 = None

        self.event_maker = EventCPU()

    def get_mean_Cp(self):
        return self.Cp

    def get_mean_Cn(self):
        return self.Cn

    def get_max_nb_events(self):
        return self.event_maker.get_max_nb_events()

    def set_config(self, config='noisy'):
        """Set configuration

        Args:
            config (str): name for configuration
        """
        if config == 'clean':
            self.Cp = 0.2
            self.Cn = 0.2
            self.sigma_threshold = 0.001
            self.cutoff_hz = 0
            self.leak_rate_hz = 0
            self.shot_noise_rate_hz = 0
        elif config == 'noisy':
            self.Cp = 0.2
            self.Cn = 0.2
            self.sigma_threshold = 0.05
            self.cutoff_hz = 70
            self.leak_rate_hz = 0.1
            self.shot_noise_rate_hz = 10
        elif config == 'intermediate':
            self.Cp = 0.2
            self.Cn = 0.2
            self.sigma_threshold = 0.01
            self.cutoff_hz = 10
            self.leak_rate_hz = 0.01
            self.shot_noise_rate_hz = 0.01

        self.Cps = self.sigma_threshold * np.random.randn(self.height, self.width).astype(np.float32) + self.Cp
        self.Cps = np.clip(self.Cps, 0.01, 1)
        self.Cns = self.sigma_threshold * np.random.randn(self.height, self.width).astype(np.float32) + self.Cn
        self.Cns = np.clip(self.Cns, 0.01, 1)

    def reset(self):
        """
        Resets buffers
        """
        self.first_pass = True
        if self.last_img is not None:
            self.last_img[...] = 0
        self.last_event_timestamp[...] = 0
        self.ref_values[...] = 0
        self.flush_events()
        self.lpLogFrame0 = None
        self.lpLogFrame1 = None
        self.last_img = None
        self.last_img_ts = 0.

    def get_size(self):
        """Function returning the size of the imager which produced the events.

        Returns:
            Tuple of int (height, width) which might be (None, None)"""
        return self.height, self.width

    def get_events(self):
        """Grab events
        """
        return self.event_maker.get_events()

    def flush_events(self):
        """Erase current events
        """
        self.event_maker.flush_events()

    def log_image_callback(self, log_img, img_ts):
        """
        For debugging, log is done outside
        """
        if self.first_pass:
            self.ref_values[...] = log_img
            self.last_img = log_img.copy()
            self.first_pass = False
            num = 0
            self.last_img_ts = img_ts
        else:
            last_img_ts = self.last_img_ts
            delta_t = img_ts - last_img_ts

            num, num_iters = self.event_maker.accumulate(self.ref_values,
                                                         self.last_img,
                                                         self.last_event_timestamp,
                                                         log_img,
                                                         last_img_ts,
                                                         delta_t,
                                                         self.Cps,
                                                         self.Cns,
                                                         self.refractory_period)

            if self.leak_rate_hz > 0:
                self.leak_events(delta_t)
            if num_iters > 0 and self.shot_noise_rate_hz > 0:
                num = self.shot_noise_events(self.event_maker.event_buffer, img_ts, num, num_iters)

            self.event_maker.num = num

            self.last_img = log_img
            self.last_img_ts = img_ts
        return num

    def image_callback(self, img, img_ts):
        """
        Accumulates Events into internal buffer

        Args:
            img (np.ndarray): uint8 gray image of shape (H,W)
            img_ts (int): timestamp in micro-seconds.

        Returns:
            num: current total number of events
        """
        assert img.shape == (self.height, self.width)
        if self.first_pass:
            log_img = self.dynamic_moving_average(img, img_ts)
            self.ref_values[...] = log_img
            self.last_img = log_img
            self.first_pass = False
            num = 0
            self.last_img_ts = img_ts
        else:
            log_img = self.dynamic_moving_average(img, img_ts)

            last_img_ts = self.last_img_ts
            delta_t = img_ts - last_img_ts

            num, num_iters = self.event_maker.accumulate(self.ref_values,
                                                         self.last_img,
                                                         self.last_event_timestamp,
                                                         log_img,
                                                         last_img_ts,
                                                         delta_t,
                                                         self.Cps,
                                                         self.Cns,
                                                         self.refractory_period)

            if self.leak_rate_hz > 0:
                self.leak_events(delta_t)
            if num_iters > 0 and self.shot_noise_rate_hz > 0:
                num = self.shot_noise_events(self.event_maker.event_buffer, img_ts, num, num_iters)

            self.event_maker.num = num

            self.last_img = log_img
            self.last_img_ts = img_ts
        return num

    def leak_events(self, delta_t):
        """
        Leak events: switch in diff change amp leaks at some rate
        equivalent to some hz of ON events.
        Actual leak rate depends on threshold for each pixel.
        We want nominal rate leak_rate_Hz, so
        R_l=(dI/dt)/Theta_on, so
        R_l*Theta_on=dI/dt, so
        dI=R_l*Theta_on*dt

        Args:
            delta_t (int): time between 2 images (us)
        """
        if self.leak_rate_hz > 0:
            delta_t_s = (delta_t * 1e-6)
            deltaLeak = delta_t_s * self.leak_rate_hz * self.Cps  # scalars
            self.ref_values -= deltaLeak  # subtract so it increases ON events

    def shot_noise_events(self, event_buffer, ts, num_events, num_iters):
        """
        NOISE: add temporal noise here by
        simple Poisson process that has a base noise rate
        self.shot_noise_rate_hz.
        If there is such noise event,
        then we output event from each such pixel

        the shot noise rate varies with intensity:
        for lowest intensity the rate rises to parameter.
        the noise is reduced by factor
        SHOT_NOISE_INTEN_FACTOR for brightest intensities

        Args:
            ts (int): timestamp
            num_events (int): current number of events
            num_iters (int): max events per pixel since last round
        """
        if self.shot_noise_rate_hz > 0:
            SHOT_NOISE_INTEN_FACTOR = 0.25

            deltaTime = (ts - self.last_img_ts) * 1e-6

            shotNoiseFactor = (
                (self.shot_noise_rate_hz / 2) * deltaTime / num_iters) * \
                ((SHOT_NOISE_INTEN_FACTOR - 1) * self.inten01 + 1)
            # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1

            rand01 = np.random.uniform(
                size=self.ref_values.shape)  # draw samples

            # probability for each pixel is
            # dt*rate*nom_thres/actual_thres.
            # That way, the smaller the threshold,
            # the larger the rate
            shotOnProbThisSample = shotNoiseFactor * np.divide(
                self.Cp, self.Cps)
            # array with True where ON noise event
            shotOnCord = rand01 > (1 - shotOnProbThisSample)

            shotOffProbThisSample = shotNoiseFactor * np.divide(
                self.Cp, self.Cps)
            # array with True where OFF noise event
            shotOffCord = rand01 < shotOffProbThisSample

            y1, x1 = np.where(shotOffCord)

            start, end = num_events, num_events + len(y1)
            if end >= len(event_buffer) - 1:
                return end
            event_buffer[start:end]['y'] = y1
            event_buffer[start:end]['x'] = x1
            event_buffer[start:end]['p'] = 0
            event_buffer[start:end]['t'] = ts

            num_events = end

            y2, x2 = np.where(shotOnCord)

            start, end = end, end + len(y2)

            event_buffer[start:end]['y'] = y2
            event_buffer[start:end]['x'] = x2
            event_buffer[start:end]['p'] = 1
            event_buffer[start:end]['t'] = ts

            num_events = end

            self.ref_values[shotOnCord] += shotOnCord[shotOnCord] * self.Cps[shotOnCord]
            self.ref_values[shotOffCord] -= shotOffCord[shotOffCord] * self.Cns[shotOffCord]

        return num_events

    def dynamic_moving_average(self, new_frame, ts, eps=1e-7):
        """
        Apply nonlinear lowpass filter here.
        Filter is 2nd order lowpass IIR
        that uses two internal state variables
        to store stages of cascaded first order RC filters.
        Time constant of the filter is proportional to
        the intensity value (with offset to deal with DN=0)

        Args:
            new_frame (np.ndarray): new image
            ts (int): new timestamp (us)
        """
        new_frame = new_frame.astype(np.float32)

        deltaTimeUs = (ts - self.last_img_ts)
        deltaTime = deltaTimeUs * 1e-6
        logNewFrame = eps_log(new_frame, eps)

        if self.lpLogFrame0 is None:
            self.lpLogFrame0 = logNewFrame.copy()
            self.lpLogFrame1 = logNewFrame.copy()

        inten01 = None
        if self.cutoff_hz > 0 or self.shot_noise_rate_hz > 0:  # will use later
            # make sure we get no zero time constants
            # limit max time constant to ~1/10 of white intensity level
            self.inten01 = (np.array(new_frame, float) + 20) / 275
        if self.cutoff_hz <= 0:
            eps = 1
        else:
            tau = (1 / (np.pi * 2 * self.cutoff_hz))
            # make the update proportional to the local intensity
            eps = self.inten01 * (deltaTime / tau)
            eps[eps[:] > 1] = 1  # keep filter stable

        # first internal state is updated
        self.lpLogFrame0 = (1 - eps) * self.lpLogFrame0 + eps * logNewFrame
        # then 2nd internal state (output) is updated from first
        self.lpLogFrame1 = (1 - eps) * self.lpLogFrame1 + eps * self.lpLogFrame0
        return self.lpLogFrame1

    def __del__(self):
        del self.event_maker
        del self.ref_values
        del self.last_img
        del self.last_event_timestamp
