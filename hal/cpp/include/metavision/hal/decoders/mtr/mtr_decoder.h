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

#ifndef METAVISION_HAL_MTR_DECODER_H
#define METAVISION_HAL_MTR_DECODER_H

#include <map>
#include <memory>
#include <string>

#include <metavision/sdk/base/events/event_pointcloud.h>
#include <metavision/hal/facilities/i_event_frame_decoder.h>
#include <metavision/hal/facilities/i_geometry.h>
#include <metavision/hal/utils/hal_log.h>

namespace Metavision {

class MTRDecoder : public I_EventFrameDecoder<PointCloud> {
public:
    enum MTRMode { MTR, MTRMetric, MTRHomogeneous, MTRScaled, MTRHomogeneousScaled };

    /// @brief Constructor
    /// @param sensor_width Width of the sensor
    /// @param sensor_height Height of the sensor
    /// @param mode MTR (Module for Triangulation) stream mode
    MTRDecoder(int sensor_width = 1280, int sensor_height = 720, MTRMode mode = MTRMode::MTR) :
        I_EventFrameDecoder(sensor_height, sensor_width), mode_(mode) {}

    virtual ~MTRDecoder() {}

    /// @brief Gets size of a raw event in bytes
    uint8_t get_raw_event_size_bytes() const override {
        switch (mode_) {
        case MTRMode::MTRMetric:
        case MTRMode::MTRHomogeneous:
        case MTRMode::MTRScaled:
        case MTRMode::MTRHomogeneousScaled:
            return sizeof(Evt_MTRU);
        default:
            return sizeof(Evt_MTR);
        }
    }

    void decode(const RawData *ev_bufferp, const RawData *evendp) override {
        switch (mode_) {
        case MTRMode::MTRMetric:
            decode_impl_MTRU(ev_bufferp, evendp, [this](uint32_t X, uint32_t Y, uint32_t Z, int ch_id, int rep_id) {
                int32_t xs             = X & (1 << 15) ? X | 0xFFFF0000 : X;
                int32_t ys             = Y & (1 << 15) ? Y | 0xFFFF0000 : Y;
                PointCloud::Point3D pt = {xs / ((float)(1 << 11)), ys / ((float)(1 << 11)), Z / ((float)(1 << 12)),
                                          ch_id, rep_id};
                return pt;
            });
            break;
        case MTRMode::MTRHomogeneous:
            decode_impl_MTRU(ev_bufferp, evendp, [](uint32_t X, uint32_t Y, uint32_t Z, int ch_id, int rep_id) {
                int32_t xs             = X & (1 << 15) ? X | 0xFFFF0000 : X;
                int32_t ys             = Y & (1 << 15) ? Y | 0xFFFF0000 : Y;
                float Zf               = Z / ((float)(1 << 12));
                PointCloud::Point3D pt = {xs * Zf / ((float)(1 << 15)), ys * Zf / ((float)(1 << 15)), Zf, ch_id,
                                          rep_id};
                return pt;
            });
            break;
        case MTRMode::MTRHomogeneousScaled:
            decode_impl_MTRU(ev_bufferp, evendp, [this](uint32_t X, uint32_t Y, uint32_t Z, int ch_id, int rep_id) {
                int32_t xs             = X & (1 << 15) ? X | 0xFFFF0000 : X;
                int32_t ys             = Y & (1 << 15) ? Y | 0xFFFF0000 : Y;
                int32_t zs             = Z & (1 << 15) ? Z | 0xFFFF0000 : Z;
                float Zf               = zs * Z_max_ / ((float)(1 << 15));
                PointCloud::Point3D pt = {xs * Zf * X_max_ / ((float)(1 << 15)), ys * Zf * Y_max_ / ((float)(1 << 15)),
                                          Zf, ch_id, rep_id};
                return pt;
            });
            break;
        case MTRMode::MTRScaled:
            decode_impl_MTRU(ev_bufferp, evendp, [this](uint32_t X, uint32_t Y, uint32_t Z, int ch_id, int rep_id) {
                int32_t xs             = X & (1 << 15) ? X | 0xFFFF0000 : X;
                int32_t ys             = Y & (1 << 15) ? Y | 0xFFFF0000 : Y;
                PointCloud::Point3D pt = {xs * X_max_ / ((float)(1 << 15)), ys * Y_max_ / ((float)(1 << 15)),
                                          Z * Z_max_ / ((float)(1 << 15)), ch_id, rep_id};
                return pt;
            });
            break;
        default:
            decode_impl_MTRPixel(ev_bufferp, evendp);
            break;
        }
    }

private:
    struct Evt_MTR {
        uint64_t Z : 19;
        uint64_t y : 10;
        uint64_t x : 15;
        uint64_t padding1 : 4;
        uint64_t channel_number : 3;
        uint64_t padding2 : 1;
        uint64_t repetition_id : 3;
        uint64_t padding3 : 1;
        uint64_t frame_number : 7;
        uint64_t padding4 : 1;
    };

    struct Evt_MTRU {
        uint32_t Z : 16;
        uint32_t Y : 16;
        uint32_t X : 16;
        uint32_t channel_number : 3;
        uint32_t padding2 : 1;
        uint32_t repetition_id : 3;
        uint32_t padding3 : 1;
        uint32_t frame_number : 7;
        uint32_t padding4 : 1;
    };

    static_assert(sizeof(Evt_MTR) == 8, "The size of the packed struct Evt_MTR is not expected one (which is 8 bytes)");
    static_assert(sizeof(Evt_MTRU) == 8,
                  "The size of the packed struct Evt_MTRU is not expected one (which is 8 bytes)");

    void decode_impl_MTRPixel(const RawData *ev_bufferp, const RawData *evendp) {
        const Evt_MTR *cur_ev = reinterpret_cast<const Evt_MTR *>(ev_bufferp);
        const Evt_MTR *ev_end = reinterpret_cast<const Evt_MTR *>(evendp);

        for (; cur_ev < ev_end; ++cur_ev) {
            const unsigned int ev_ch_num    = cur_ev->channel_number;
            const unsigned int ev_frame_num = cur_ev->frame_number;
            // Check first if we are changing channel
            if (ev_ch_num != cur_channel_id_) {
                // Check correct increment
                if (ev_ch_num < cur_channel_id_ && ev_frame_num == cur_frame_counter_) {
                    MV_HAL_LOG_WARNING() << "Moving from channel " << cur_channel_id_ << " to channel " << ev_ch_num
                                         << " in the same frame." << std::endl;
                }

                cur_channel_id_ = ev_ch_num;
            }
            // If we also changed frame, let's call the correct callback
            if (ev_frame_num != cur_frame_counter_) {
                // Change of frame detected, check correct increment
                if (ev_frame_num - cur_frame_counter_ != 1 && !(ev_frame_num == 0 && cur_frame_counter_ == 0x07F)) {
                    MV_HAL_LOG_WARNING() << "Moving from frame " << cur_frame_counter_ << " to frame " << ev_frame_num
                                         << std::endl;
                }
                // Keep track of new info
                cur_frame_counter_ = ev_frame_num;
                ++cur_frame_id_;
                // Callbacks
                add_event_frame({cur_frame_id_, Z_max_, std::move(pointcloud_)});
            }
            // Accumulate data present in event
            pointcloud_.push_back({cur_ev->x / 16.f, (float)cur_ev->y, cur_ev->Z / ((float)(1 << 15)),
                                   (int)cur_ev->channel_number, (int)cur_ev->repetition_id});
        }
    }

    void decode_impl_MTRU(const RawData *ev_bufferp, const RawData *evendp,
                          const std::function<PointCloud::Point3D(uint32_t X, uint32_t Y, uint32_t Z, int ch_id,
                                                                  int rep_id)> &Transform) {
        const Evt_MTRU *cur_ev = reinterpret_cast<const Evt_MTRU *>(ev_bufferp);
        const Evt_MTRU *ev_end = reinterpret_cast<const Evt_MTRU *>(evendp);

        for (; cur_ev < ev_end; ++cur_ev) {
            const unsigned int ev_ch_num    = cur_ev->channel_number;
            const unsigned int ev_frame_num = cur_ev->frame_number;
            // Check first if we are changing channel
            if (ev_ch_num != cur_channel_id_) {
                // Check correct increment
                if (ev_ch_num < cur_channel_id_ && ev_frame_num == cur_frame_counter_) {
                    MV_HAL_LOG_WARNING() << "Error: Moving from channel " << cur_channel_id_ << " to channel "
                                         << ev_ch_num << " in the same frame." << std::endl;
                }

                cur_channel_id_ = ev_ch_num;
            }
            // If we also changed frame, let's call the correct callback
            if (ev_frame_num != cur_frame_counter_) {
                // Change of frame detected, check correct increment
                if (ev_frame_num - cur_frame_counter_ != 1 && !(ev_frame_num == 0 && cur_frame_counter_ == 0x07F)) {
                    MV_HAL_LOG_WARNING() << "Error: Moving from frame " << cur_frame_counter_ << " to frame "
                                         << ev_frame_num << std::endl;
                }
                // Keep track of new info
                cur_frame_counter_ = ev_frame_num;
                ++cur_frame_id_;
                // Callbacks
                add_event_frame({cur_frame_id_, Z_max_, std::move(pointcloud_)});
            }
            // Accumulate data present in event
            pointcloud_.emplace_back(
                Transform(cur_ev->X, cur_ev->Y, cur_ev->Z, cur_ev->channel_number, cur_ev->repetition_id));
        }
    }

    unsigned int cur_frame_id_      = 0;
    unsigned int cur_frame_counter_ = 0;
    unsigned int cur_channel_id_    = 0;
    std::vector<PointCloud::Point3D> pointcloud_;
    MTRMode mode_;

    const float X_max_ = 16.f, Y_max_ = 16.f, Z_max_ = 16.f;
};

inline std::unique_ptr<MTRDecoder> mtr_decoder_from_format(const I_Geometry &geometry, const std::string &format) {
    static std::map<std::string, MTRDecoder::MTRMode> format_to_mode = {
        {"MTR", MTRDecoder::MTRMode::MTR},
        {"MTRMetric", MTRDecoder::MTRMode::MTRMetric},
        {"MTRHomogeneous", MTRDecoder::MTRMode::MTRHomogeneous},
        {"MTRScaled", MTRDecoder::MTRMode::MTRScaled},
        {"MTRHomogeneousScaled", MTRDecoder::MTRMode::MTRHomogeneousScaled}};

    if (format_to_mode.count(format) <= 0) {
        return nullptr;
    }

    return std::make_unique<MTRDecoder>(geometry.get_width(), geometry.get_height(), format_to_mode[format]);
}

} // namespace Metavision

#endif /* METAVISION_HAL_MTR_DECODER_H */
