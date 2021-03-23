/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#ifndef METAVISION_PLAYER_ANALYSIS_UTILS_H
#define METAVISION_PLAYER_ANALYSIS_UTILS_H

#include <metavision/sdk/base/utils/timestamp.h>

#include "utils.h"
#include "analysis_view.h"

inline Metavision::timestamp compute_sequence_start_time(Metavision::timestamp first_time_us,
                                                         Metavision::timestamp last_time_us, int frame_period_us,
                                                         int sequence_start_ratio) {
    const Metavision::timestamp min_start_time_us = first_time_us + frame_period_us;
    Metavision::timestamp max_start_time_us       = min_start_time_us;
    while (max_start_time_us <= last_time_us) {
        max_start_time_us += frame_period_us;
    }
    const Metavision::timestamp start_time_us = static_cast<Metavision::timestamp>(
        sequence_start_ratio * (max_start_time_us - min_start_time_us) / 100. + min_start_time_us + 0.5);
    return clip(start_time_us, min_start_time_us, max_start_time_us);
}

inline Metavision::timestamp compute_sequence_duration(Metavision::timestamp first_time_us,
                                                       Metavision::timestamp last_time_us, int frame_period_us,
                                                       int sequence_start_ratio, int sequence_duration_ratio) {
    const Metavision::timestamp start_time_us =
        compute_sequence_start_time(first_time_us, last_time_us, frame_period_us, sequence_start_ratio);
    Metavision::timestamp time_us         = start_time_us + frame_period_us;
    Metavision::timestamp max_duration_us = frame_period_us;
    while (time_us <= last_time_us) {
        time_us += frame_period_us;
        max_duration_us += frame_period_us;
    }
    const Metavision::timestamp min_duration_us = frame_period_us;
    Metavision::timestamp duration_us           = static_cast<Metavision::timestamp>(
        sequence_duration_ratio * (max_duration_us - min_duration_us) / 100. + 0.5 + min_duration_us);
    duration_us =
        static_cast<Metavision::timestamp>((duration_us + frame_period_us - 1.0) / frame_period_us) * frame_period_us;
    return clip(duration_us, min_duration_us, max_duration_us);
}

inline int compute_current_time(Metavision::timestamp sequence_start_time_us, int frame_id, int frame_period_us) {
    return sequence_start_time_us + frame_id * frame_period_us;
}

struct AnalysisData {
    int fps, min_fps, max_fps;
    int min_sequence_start_ratio, max_sequence_start_ratio;
    int min_sequence_duration_ratio, max_sequence_duration_ratio;
    int accumulation_ratio, min_accumulation_ratio, max_accumulation_ratio;
    int frame_id, min_frame_id, max_frame_id;
};

inline AnalysisData compute_analysis_data(Metavision::timestamp first_time_us, Metavision::timestamp last_time_us,
                                          int fps, int accumulation_ratio, int sequence_start_ratio,
                                          int sequence_duration_ratio, int current_time_us) {
    AnalysisData data;

    int frame_period_us = compute_frame_period(fps);
    const Metavision::timestamp sequence_start_time_us =
        compute_sequence_start_time(first_time_us, last_time_us, frame_period_us, sequence_start_ratio);
    const Metavision::timestamp sequence_duration_us = compute_sequence_duration(
        first_time_us, last_time_us, frame_period_us, sequence_start_ratio, sequence_duration_ratio);
    const Metavision::timestamp accumulation_time_us = compute_accumulation_time(accumulation_ratio, frame_period_us);

    // Update fps given start time
    const Metavision::timestamp min_frame_period_us = 1;
    const Metavision::timestamp max_frame_period_us = last_time_us - first_time_us;
    const int min_fps =
        clip(static_cast<int>(1.e6 / max_frame_period_us + 0.5), AnalysisView::MinFps(), AnalysisView::MaxFps());
    const int max_fps =
        clip(static_cast<int>(1.e6 / min_frame_period_us + 0.5), AnalysisView::MinFps(), AnalysisView::MaxFps());

    data.fps        = clip(fps, min_fps, max_fps);
    data.max_fps    = max_fps;
    data.min_fps    = min_fps;
    frame_period_us = compute_frame_period(data.fps);

    // Update start ratio
    data.max_sequence_start_ratio = 100;
    data.min_sequence_start_ratio = 0;

    // Update duration ratio
    data.max_sequence_duration_ratio = 100;
    data.min_sequence_duration_ratio = 1;

    // Update accumulation ratio given start time and frame period : we cannot accumulate more than
    // sequence_start_time_us in any case
    const int max_accumulation_ratio = clip(static_cast<int>(sequence_start_time_us * 100. / frame_period_us + 0.5),
                                            AnalysisView::MinAccumulationRatio(), AnalysisView::MaxAccumulationRatio());
    data.accumulation_ratio = clip(accumulation_ratio, AnalysisView::MinAccumulationRatio(), max_accumulation_ratio);
    data.max_accumulation_ratio = max_accumulation_ratio;
    data.min_accumulation_ratio = AnalysisView::MinAccumulationRatio();

    // Update frame id given start time and frame duration
    const Metavision::timestamp sequence_end_time_us = sequence_start_time_us + sequence_duration_us;
    Metavision::timestamp time_us                    = sequence_start_time_us;
    Metavision::timestamp max_frame_id               = 0;
    while (time_us + frame_period_us < sequence_end_time_us) {
        time_us += frame_period_us;
        ++max_frame_id;
    }
    data.max_frame_id = max_frame_id;
    data.min_frame_id = 0;
    data.frame_id     = (current_time_us - sequence_start_time_us + frame_period_us - 1) / frame_period_us;

    return data;
}

#endif // METAVISION_PLAYER_ANALYSIS_UTILS_H
