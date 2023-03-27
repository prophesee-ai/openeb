# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# pylint: disable=E0633
# pylint: disable=E1121
"""
Implementation in numba cuda GPU of kernels used to simulate events from images
GPU kernels used to simulate events from images
"""
from __future__ import absolute_import

import math
from copy import deepcopy

from numba import cuda, jit


exec_string = """@{decorator}
def _{runtime}_kernel_{func_name}(
        {params}, log_sequence, num_frames_cumsum, image_times, first_times, rng_states, log_states, prev_log_images,
        timestamps, thresholds, previous_image_times, refractory_periods, leak_rates, shot_noise_rates, threshold_mus,
        persistent=True, {default_params}):
    '''
    {documentation}
    '''
    height, width = log_sequence.shape[:2]
    batch_size = len(num_frames_cumsum)
    {loop}
                    last_timestamp_at_xy = 0 if first_time else timestamps[b, y, x]
                    log_state_at_xy = log_states[b, y, x]
                    Cp = thresholds[1, b, y, x]
                    Cn = thresholds[0, b, y, x]
                    end_f = num_frames_cumsum[b]
                    start_f = num_frames_cumsum[b-1] if b else 0
                    {on_start}

                    for tt in range(start_f, end_f):

                        if first_time and tt == start_f:
                            log_state_at_xy = log_sequence[y, x, start_f]
                            continue
                        elif tt == start_f:
                            it = prev_log_images[b, y, x]
                            last_image_ts = previous_image_times[b]
                        else:
                            it = log_sequence[y, x, tt-1]
                            last_image_ts = image_times[b, tt-start_f-1]

                        curr_image_ts = image_times[b, tt- start_f]

                        itdt = log_sequence[y, x, tt]
                        prev_ref_val = log_state_at_xy
                        pol = 1. if itdt >= it else -1.
                        p = 1 if itdt >= it else 0
                        C = thresholds[p, b, y, x]
                        delta_t = curr_image_ts - last_image_ts
                        all_crossing = False
                        polC = pol * C
                        num_events = 0

                        # Better Refractory Model!
                        # TODO: during characterization story MV-67, check if this code
                        # improves realism
                        # dt_since_last_event = curr_image_ts - last_timestamp_at_xy
                        # if dt_since_last_event >= refractory_period and last_timestamp_at_xy > 0:
                        #     time_end_refractory_period = last_timestamp_at_xy + refractory_period
                        #     if last_image_ts <= time_end_refractory_period <= curr_image_ts:
                        #         dt_since_last_image = time_end_refractory_period - last_image_ts
                        #         ratio = dt_since_last_image / delta_t
                        #         prev_ref_val = it * (1-ratio) + itdt * ratio
                        #         it = prev_ref_val

                        #         last_image_ts = time_end_refractory_period

                        if abs(itdt - prev_ref_val) > C:
                            current_ref_val = prev_ref_val

                            while not all_crossing:
                                current_ref_val += polC

                                if (pol > 0 and current_ref_val > it and current_ref_val <= itdt) \
                                        or  (pol < 0 and current_ref_val < it and current_ref_val >= itdt):
                                    edt = (current_ref_val - it) * delta_t / (itdt - it)
                                    ts = int(last_image_ts + edt)
                                    dt = ts - last_timestamp_at_xy
                                    if dt >= refractory_period or last_timestamp_at_xy == 0:
                                        num_events += 1
                                        last_timestamp_at_xy = ts

                                        {on_event_write}

                                    log_state_at_xy = current_ref_val
                                else:
                                    all_crossing = True

                        it = itdt

                        # shot noise
                        if shot_noise_micro_hz > 0:
                            intensity = math.exp(itdt)
                            shot_noise_factor = (shot_noise_micro_hz / 2) * delta_t / (1 + num_events)
                            shot_noise_factor *= (-0.75 * intensity + 1)
                            shot_on_prob = shot_noise_factor * threshold_mu[1] / thresholds[1, b, y, x]
                            shot_off_prob = shot_noise_factor * threshold_mu[0] / thresholds[0, b, y, x]
                            rand_on = rng_states[b, y, x] * (math.sin(curr_image_ts) + 1) / 2
                            rand_off = rng_states[b, y, x] * (math.cos(curr_image_ts) + 1) / 2
                            if rand_on > (1 - shot_on_prob):
                                pol = 1
                                ts = curr_image_ts
                                log_state_at_xy += Cp
                                last_timestamp_at_xy = ts
                                {on_event_write}
                            if rand_off > (1 - shot_off_prob):
                                pol = -1
                                ts = curr_image_ts
                                last_timestamp_at_xy = ts
                                log_state_at_xy -= Cn
                                {on_event_write}

                        # noise leak-rate
                        deltaLeak = delta_t * leak_rate_micro_hz * Cp
                        log_state_at_xy -= deltaLeak

                    if persistent:
                        timestamps[b, y, x] = last_timestamp_at_xy
                        log_states[b, y, x] = log_state_at_xy
                        prev_log_images[b, y, x] = log_sequence[y, x, end_f - 1]

        """


def loop(runtime):
    if runtime == "cuda":
        return """
    b, y, x = cuda.grid(3)
    if b < batch_size and y < height and x < width:
                    first_time = int(first_times[b])
                    refractory_period = refractory_periods[b]
                    leak_rate_micro_hz = leak_rates[b]
                    shot_noise_micro_hz = shot_noise_rates[b]
                    threshold_mu = threshold_mus[b]"""
    elif runtime == "cpu":
        return """
    for b in range(batch_size):
        refractory_period = refractory_periods[b]
        leak_rate_micro_hz = leak_rates[b]
        shot_noise_micro_hz = shot_noise_rates[b]
        threshold_mu = threshold_mus[b]
        first_time = int(first_times[b])
        for y in range(height):
            for x in range(width):"""
    else:
        raise ValueError(f"unsupported runtime {runtime}")


def format_kernel_string(func_name="", params="", default_params="", runtime='cuda',
                         on_event_write="", documentation="", on_start=""):
    loop_text = loop(runtime)
    decorator = "cuda.jit()" if runtime == "cuda" else "jit(nopython=True)"
    if "{runtime}" in on_event_write:
        on_event_write = on_event_write.format(runtime=runtime)
    return exec_string.format(func_name=func_name, params=params, default_params=default_params, loop=loop_text,
                              runtime=runtime, on_event_write=on_event_write, documentation=documentation,
                              on_start=on_start, decorator=decorator)


def format_kernel_strings(func_name="", params="", default_params="", runtimes=('cuda', "cpu"),
                          on_event_write="", documentation="", on_start=""):
    return "\n".join(format_kernel_string(func_name=func_name, params=params, default_params=default_params,
                                          runtime=runtime, on_event_write=on_event_write,
                                          documentation=documentation, on_start=on_start) for runtime in runtimes)
#######################################################################################################################


exec(format_kernel_strings(func_name="count_events", params="counts", on_event_write="counts[b, y, x] += 1",
                           documentation="Counts num_events / pixel "))

#######################################################################################################################

exec(format_kernel_strings(
    func_name="fill_events", params="events, offsets", on_start="index = offsets[b, y, x]",
    on_event_write="events[index, 0] = int(b); events[index, 1] = int(x); events[index, 2] = int(y);"
                   " events[index, 3] = pol; events[index, 4] = ts; index += 1",
    documentation="Fills an event-buffer "))


#######################################################################################################################


def fill_voxel_sequence(b, ts, x, y, pol, nbins, target_times, bin_index, voxel_grid, bilinear, split):
    if split:
        num_bins = nbins//2
    else:
        num_bins = nbins
    # fills voxel grid non linear cuts
    while ts > target_times[b, bin_index + 1] and bin_index < len(voxel_grid)-1:
        bin_index += 1

    bin_index = min(bin_index, len(voxel_grid)-1)
    t0 = target_times[b, bin_index]
    tn = target_times[b, bin_index + 1]
    dt = tn - t0
    ti_star = ((ts - t0) * num_bins / dt) - 0.5
    lbin = int(math.floor(ti_star))
    lbin = max(lbin, 0)
    lbin = min(lbin, num_bins-1)
    if split:
        if pol > 0:
            lbin = lbin + num_bins
        pol = 1.0
    if bilinear:
        rbin = min(lbin + 1, num_bins - 1)
        left_value = max(0, 1 - abs(lbin - ti_star))
        right_value = 1 - left_value

        if 0 <= lbin < nbins:
            voxel_grid[bin_index, b, lbin, y, x] += float(left_value * pol)
        if 0 <= rbin < nbins:
            voxel_grid[bin_index, b, rbin, y, x] += float(right_value * pol)
    else:
        if 0 <= lbin < nbins:
            voxel_grid[bin_index, b, lbin, y, x] += pol


fill_voxel_sequence_cuda = cuda.jit(device=True)(fill_voxel_sequence)
fill_voxel_sequence_cpu = jit(nopython=True)(fill_voxel_sequence)

exec(format_kernel_strings(
    func_name="voxel_grid_sequence", params="voxel_grid, target_times",
    default_params="bilinear=True, split=False",
    on_event_write="fill_voxel_sequence_{runtime}(b, ts, x, y, pol, nbins, target_times, bin_index, voxel_grid, bilinear, split)",
    on_start="nbins = voxel_grid.shape[2]; t0 = target_times[b, 0]; bin_index = 0",
    documentation="Computes an event cube sequence"))
