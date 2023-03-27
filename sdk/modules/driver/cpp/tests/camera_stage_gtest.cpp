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

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <numeric>
#include <array>
#include <boost/filesystem.hpp>

#include "metavision/sdk/base/events/event_cd.h"
#include "metavision/utils/gtest/gtest_with_tmp_dir.h"
#include "metavision/sdk/driver/camera_exception.h"
#include "metavision/sdk/driver/pipeline/camera_stage.h"
#include "metavision/sdk/driver/internal/camera_internal.h"
#include "metavision/sdk/core/pipeline/pipeline.h"
#include "metavision/hal/facilities/i_events_stream_decoder.h"
#include "metavision/hal/utils/raw_file_header.h"
#include "tencoder_gtest_common.h"

using namespace Metavision;

class CameraStage_Gtest : public GTestWithTmpDir {
protected:
    virtual void SetUp() override {
        // Create and open the RAW file
        static int raw_counter = 1;
        rawfile_to_log_path_   = tmpdir_handler_->get_full_path("rawfile_" + std::to_string(++raw_counter) + ".raw");
        encoded_bytes_         = 0;
    }

    std::vector<EventCD> write_ref_data() {
        std::ofstream rawfile_to_log(rawfile_to_log_path_, std::ios::out | std::ios::binary);
        EXPECT_TRUE(rawfile_to_log.is_open());

        RawFileHeader header_to_write;
        header_to_write.set_plugin_name(dummy_plugin_name_);
        header_to_write.set_camera_integrator_name(dummy_camera_integrator_name_);
        header_to_write.set_plugin_integrator_name(dummy_plugin_integrator_name_);
        header_to_write.set_field(dummy_custom_key_, dummy_custom_value_);

        // Prophesee header only. Duplicated what PropheseeRawHeader does to be able to encode then read test RAW
        // file
        header_to_write.set_field("system_ID", std::to_string(gen31_system_id));

        rawfile_to_log << header_to_write;

        auto events = build_vector_of_events<Evt2RawFormat, EventCD>();
        TEncoder<Evt2RawFormat, TimerHighRedundancyEvt2Default> encoder;
        encoder.set_encode_event_callback([&](const uint8_t *data, const uint8_t *data_end) {
            encoded_bytes_ += std::distance(data, data_end);
            rawfile_to_log.write(reinterpret_cast<const char *>(data), std::distance(data, data_end));
        });

        encoder.encode(events.cbegin(), events.cend());
        encoder.flush();
        return events;
    }

public:
    static const std::string dummy_plugin_name_;
    static const std::string dummy_camera_integrator_name_;
    static const std::string dummy_plugin_integrator_name_;
    static const std::string dummy_custom_key_;
    static const std::string dummy_custom_value_;

    static constexpr long gen31_system_id = 28;

protected:
    std::string rawfile_to_log_path_;
    size_t encoded_bytes_;
};

constexpr long CameraStage_Gtest::gen31_system_id;
const std::string CameraStage_Gtest::dummy_plugin_name_            = "hal_plugin_gen31_fx3";
const std::string CameraStage_Gtest::dummy_camera_integrator_name_ = "Prophesee";
const std::string CameraStage_Gtest::dummy_plugin_integrator_name_ = "Prophesee";
const std::string CameraStage_Gtest::dummy_custom_key_             = "custom";
const std::string CameraStage_Gtest::dummy_custom_value_           = "field";

struct MockStage : public BaseStage {
public:
    MockStage() {
        set_consuming_callback([this](const boost::any &data) {
            auto events = boost::any_cast<EventBufferPtr>(data);
            events_received_.insert(events_received_.end(), events->cbegin(), events->cend());
        });
    }

    std::vector<EventCD> events_received_;
};

TEST_F(CameraStage_Gtest, camera_stage_produces_correct_events) {
    ////////////////////////////////////////////////////////////////////////////////
    // PURPOSE
    // Checks that the camera stage produces correct events

    std::vector<EventCD> ref_data = write_ref_data();

    Camera camera        = Camera::from_file(rawfile_to_log_path_);
    size_t decoded_bytes = 0;
    camera.raw_data().add_callback([&](auto, size_t bytes_count) { decoded_bytes += bytes_count; });
    Pipeline p(true);
    auto &cam_stage = p.add_stage(std::make_unique<CameraStage>(std::move(camera)));
    cam_stage.detach();
    auto &mock_stage = p.add_stage(std::make_unique<MockStage>(), cam_stage);
    mock_stage.detach();

    p.run();

    ASSERT_EQ(encoded_bytes_, decoded_bytes);
    ASSERT_EQ(ref_data.size(), mock_stage.events_received_.size());

    timestamp tshift;
    {
        ASSERT_NO_THROW(cam_stage.camera().get_device());
        auto &device        = cam_stage.camera().get_device();
        const auto &decoder = device.get_facility<I_EventsStreamDecoder>();

        ASSERT_TRUE(decoder);

        EXPECT_TRUE(decoder->get_timestamp_shift(tshift));
    }

    using SizeType = std::vector<EventCD>::size_type;
    for (SizeType i = 0; i < ref_data.size(); ++i) {
        EXPECT_EQ(ref_data[i].x, mock_stage.events_received_[i].x);
        EXPECT_EQ(ref_data[i].y, mock_stage.events_received_[i].y);
        EXPECT_EQ(ref_data[i].t, mock_stage.events_received_[i].t + tshift);
        EXPECT_EQ(ref_data[i].p, mock_stage.events_received_[i].p);
    }
}
