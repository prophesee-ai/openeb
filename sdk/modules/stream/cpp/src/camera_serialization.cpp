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

#ifdef HAS_PROTOBUF
#include <sstream>
#include <variant>
#include <google/protobuf/util/json_util.h>

#include "metavision/hal/facilities/i_antiflicker_module.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/facilities/i_digital_crop.h"
#include "metavision/hal/facilities/i_digital_event_mask.h"
#include "metavision/hal/facilities/i_erc_module.h"
#include "metavision/hal/facilities/i_event_rate_activity_filter_module.h"
#include "metavision/hal/facilities/i_event_trail_filter_module.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "metavision/hal/facilities/i_ll_biases.h"
#include "metavision/hal/facilities/i_roi.h"
#include "metavision/hal/facilities/i_trigger_in.h"
#include "metavision/hal/facilities/i_trigger_out.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/sdk/base/utils/sdk_log.h"
#include "metavision/sdk/stream/camera_exception.h"
#include "metavision/sdk/stream/camera_error_code.h"
#include "metavision/sdk/stream/internal/camera_serialization.h"
#include "device_state.pb.h"
#endif // HAS_PROTOBUF

#include "metavision/hal/device/device.h"

namespace Metavision {

#ifdef HAS_PROTOBUF

template<class... Facilities>
struct FacilityTypeList {
    using PtrVariant      = std::variant<Facilities *...>;
    using ConstPtrVariant = std::variant<const Facilities *...>;

    static std::vector<ConstPtrVariant> get_facilities(const Device &device) {
        return {device.get_facility<Facilities>()...};
    }

    static std::vector<PtrVariant> get_facilities(Device &device) {
        return {device.get_facility<Facilities>()...};
    }
};

// Note : I_HW_Register should be the last item in this list so that it is deserialized at the end
// This way, it can be documented that registers will not potentially be overwritten by the
// deserialization of other facilities
using FacilityTypes = FacilityTypeList<I_AntiFlickerModule, I_CameraSynchronization, I_DigitalCrop, I_DigitalEventMask,
                                       I_ErcModule, I_EventRateActivityFilterModule, I_EventTrailFilterModule,
                                       I_LL_Biases, I_ROI, I_TriggerIn, I_TriggerOut, I_HW_Register>;

auto get_serializable_facilities(const Device &device) {
    return FacilityTypes::get_facilities(device);
}

auto get_serializable_facilities(Device &device) {
    return FacilityTypes::get_facilities(device);
}

class FacilitySerializer {
public:
    FacilitySerializer(DeviceSerialization::DeviceState &state) : state_(state) {}

    void operator()(const I_AntiFlickerModule *module) {
        if (!module) {
            return;
        }

        auto *afk_state = state_.mutable_afk_state();
        afk_state->set_enabled(module->is_enabled());
        afk_state->set_band_low_freq(module->get_band_low_frequency());
        afk_state->set_band_high_freq(module->get_band_high_frequency());
        afk_state->set_filtering_mode(module->get_filtering_mode() == I_AntiFlickerModule::AntiFlickerMode::BAND_PASS ?
                                          DeviceSerialization::AntiflickerState::BAND_PASS :
                                          DeviceSerialization::AntiflickerState::BAND_STOP);
        afk_state->set_duty_cycle(module->get_duty_cycle());
        afk_state->set_start_threshold(module->get_start_threshold());
        afk_state->set_stop_threshold(module->get_stop_threshold());
    }

    void operator()(const I_CameraSynchronization *module) {
        if (!module) {
            return;
        }

        auto *cam_sync_state = state_.mutable_cam_sync_state();
        switch (module->get_mode()) {
        case I_CameraSynchronization::SyncMode::MASTER:
            cam_sync_state->set_sync_mode(DeviceSerialization::CameraSynchronizationState::MASTER);
            break;
        case I_CameraSynchronization::SyncMode::SLAVE:
            cam_sync_state->set_sync_mode(DeviceSerialization::CameraSynchronizationState::SLAVE);
            break;
        case I_CameraSynchronization::SyncMode::STANDALONE:
            cam_sync_state->set_sync_mode(DeviceSerialization::CameraSynchronizationState::STANDALONE);
            break;
        }
    }

    void operator()(const I_DigitalCrop *module) {
        if (!module) {
            return;
        }

        auto *digital_crop_state = state_.mutable_digital_crop_state();
        digital_crop_state->set_enabled(module->is_enabled());
        digital_crop_state->mutable_region()->set_x1(std::get<0>(module->get_window_region()));
        digital_crop_state->mutable_region()->set_y1(std::get<1>(module->get_window_region()));
        digital_crop_state->mutable_region()->set_x2(std::get<2>(module->get_window_region()));
        digital_crop_state->mutable_region()->set_y2(std::get<3>(module->get_window_region()));
    }

    void operator()(const I_DigitalEventMask *module) {
        if (!module) {
            return;
        }

        auto *digital_event_mask_state = state_.mutable_digital_event_mask_state();
        digital_event_mask_state->clear_mask();
        for (const auto &mask_ptr : module->get_pixel_masks()) {
            auto *event_mask_state = digital_event_mask_state->add_mask();
            event_mask_state->set_x(std::get<0>(mask_ptr->get_mask()));
            event_mask_state->set_y(std::get<1>(mask_ptr->get_mask()));
            event_mask_state->set_enabled(std::get<2>(mask_ptr->get_mask()));
        }
    }

    void operator()(const I_ErcModule *module) {
        if (!module) {
            return;
        }

        auto *erc_state = state_.mutable_event_rate_control_state();
        erc_state->set_enabled(module->is_enabled());
        erc_state->set_cd_event_count(module->get_cd_event_count());
    }

    void operator()(const I_EventRateActivityFilterModule *module) {
        if (!module) {
            return;
        }

        const auto supported_thresholds = module->is_thresholds_supported();
        const auto thresholds = module->get_thresholds();
        int num_supported = 0;
        {
            auto *nfl_state = state_.mutable_event_rate_activity_filter_state();
            nfl_state->set_enabled(module->is_enabled());

            if (supported_thresholds.lower_bound_start == 1) {
                ++num_supported;
                nfl_state->set_lower_start_rate_threshold(thresholds.lower_bound_start);
            }
            if (supported_thresholds.lower_bound_stop == 1) {
                ++num_supported;
                nfl_state->set_lower_stop_rate_threshold(thresholds.lower_bound_stop);
            }
            if (supported_thresholds.upper_bound_start == 1) {
                ++num_supported;
                nfl_state->set_upper_start_rate_threshold(thresholds.upper_bound_start);
            }
            if (supported_thresholds.upper_bound_stop == 1) {
                ++num_supported;
                nfl_state->set_upper_stop_rate_threshold(thresholds.upper_bound_stop);
            }
        }

        // Serialize state usable by metavision < 4.6.0
        if (num_supported == 1) {
            auto *nfl_state = state_.mutable_event_rate_noise_filter_state();
            nfl_state->set_enabled(module->is_enabled());
            if (supported_thresholds.lower_bound_start == 1) {
                nfl_state->set_event_rate_threshold(thresholds.lower_bound_start);
            } else if (supported_thresholds.lower_bound_stop == 1) {
                nfl_state->set_event_rate_threshold(thresholds.lower_bound_stop);
            } else if (supported_thresholds.upper_bound_start == 1) {
                nfl_state->set_event_rate_threshold(thresholds.upper_bound_start);
            } else if (supported_thresholds.upper_bound_stop == 1) {
                nfl_state->set_event_rate_threshold(thresholds.upper_bound_stop);
            }
        }
    }

    void operator()(const I_EventTrailFilterModule *module) {
        if (!module) {
            return;
        }

        auto *etf_state = state_.mutable_event_trail_filter_state();
        etf_state->set_enabled(module->is_enabled());
        switch (module->get_type()) {
        case I_EventTrailFilterModule::Type::TRAIL:
            etf_state->set_filtering_type(DeviceSerialization::EventTrailFilterState::TRAIL);
            break;
        case I_EventTrailFilterModule::Type::STC_CUT_TRAIL:
            etf_state->set_filtering_type(DeviceSerialization::EventTrailFilterState::STC_CUT_TRAIL);
            break;
        case I_EventTrailFilterModule::Type::STC_KEEP_TRAIL:
            etf_state->set_filtering_type(DeviceSerialization::EventTrailFilterState::STC_KEEP_TRAIL);
            break;
        }
        etf_state->set_threshold(module->get_threshold());
    }

    void operator()(const I_HW_Register *module) {
        if (!module) {
            return;
        }
    }

    void operator()(const I_LL_Biases *module) {
        if (!module) {
            return;
        }

        auto *ll_biases_state = state_.mutable_ll_biases_state();
        for (auto &bias_pair : module->get_all_biases()) {
            LL_Bias_Info info;
            module->get_bias_info(bias_pair.first, info);
            auto *bias_state = ll_biases_state->add_bias();
            bias_state->set_name(bias_pair.first);
            bias_state->set_value(bias_pair.second);
        }
    }

    void operator()(const I_ROI *module) {
        if (!module) {
            return;
        }

        auto roi_state = state_.mutable_roi_state();

        roi_state->set_enabled(module->is_enabled());
        roi_state->set_mode(module->get_mode() == I_ROI::Mode::ROI ? DeviceSerialization::RegionOfInterestState::ROI :
                                                                     DeviceSerialization::RegionOfInterestState::RONI);

        for (auto &w : module->get_windows()) {
            auto window_state = roi_state->add_window();
            window_state->set_x(w.x);
            window_state->set_width(w.width);
            window_state->set_y(w.y);
            window_state->set_height(w.height);
        }

        std::vector<bool> cols;
        std::vector<bool> rows;
        if (module->get_lines(cols, rows)) {
            for (bool c : cols) {
                roi_state->add_columns(c);
            }
            for (bool r : rows) {
                roi_state->add_rows(r);
            }
        }
    }

    void operator()(const I_TriggerIn *module) {
        if (!module) {
            return;
        }

        auto *trigger_in_state = state_.mutable_trigger_in_state();
        for (auto &channel_pair : module->get_available_channels()) {
            auto *channel_status = trigger_in_state->add_channel_status();
            switch (channel_pair.first) {
            case I_TriggerIn::Channel::Main:
                channel_status->set_channel(DeviceSerialization::TriggerInState::MAIN);
                break;
            case I_TriggerIn::Channel::Aux:
                channel_status->set_channel(DeviceSerialization::TriggerInState::AUX);
                break;
            case I_TriggerIn::Channel::Loopback:
                channel_status->set_channel(DeviceSerialization::TriggerInState::LOOPBACK);
                break;
            }
            channel_status->set_enabled(module->is_enabled(channel_pair.first));
        }
    }

    void operator()(const I_TriggerOut *module) {
        if (!module) {
            return;
        }

        auto *trigger_out_state = state_.mutable_trigger_out_state();
        trigger_out_state->set_enabled(module->is_enabled());
        trigger_out_state->set_period(module->get_period());
        trigger_out_state->set_duty_cycle(module->get_duty_cycle());
    }

private:
    DeviceSerialization::DeviceState &state_;
};

class FacilityDeserializer {
public:
    FacilityDeserializer(const DeviceSerialization::DeviceState &state) : state_(state) {}

    void operator()(I_AntiFlickerModule *module) {
        if (!module) {
            return;
        }

        if (!state_.has_afk_state()) {
            return;
        }

        const auto &afk_state = state_.afk_state();
        if (afk_state.optional_enabled_case() == DeviceSerialization::AntiflickerState::kEnabled) {
            module->enable(afk_state.enabled());
        }
        if (afk_state.optional_band_low_freq_case() == DeviceSerialization::AntiflickerState::kBandLowFreq) {
            module->set_frequency_band(afk_state.band_low_freq(), module->get_band_high_frequency());
        }
        if (afk_state.optional_band_high_freq_case() == DeviceSerialization::AntiflickerState::kBandHighFreq) {
            module->set_frequency_band(module->get_band_low_frequency(), afk_state.band_high_freq());
        }
        if (afk_state.optional_filtering_mode_case() == DeviceSerialization::AntiflickerState::kFilteringMode) {
            module->set_filtering_mode(afk_state.filtering_mode() == DeviceSerialization::AntiflickerState::BAND_PASS ?
                                           I_AntiFlickerModule::AntiFlickerMode::BAND_PASS :
                                           I_AntiFlickerModule::AntiFlickerMode::BAND_STOP);
        }
        if (afk_state.optional_duty_cycle_case() == DeviceSerialization::AntiflickerState::kDutyCycle) {
            module->set_duty_cycle(afk_state.duty_cycle());
        }
        if (afk_state.optional_start_threshold_case() == DeviceSerialization::AntiflickerState::kStartThreshold) {
            module->set_start_threshold(afk_state.start_threshold());
        }
        if (afk_state.optional_stop_threshold_case() == DeviceSerialization::AntiflickerState::kStopThreshold) {
            module->set_stop_threshold(afk_state.stop_threshold());
        }
    }

    void operator()(I_CameraSynchronization *module) {
        if (!module) {
            return;
        }

        if (!state_.has_cam_sync_state()) {
            return;
        }

        const auto &cam_sync_state = state_.cam_sync_state();
        if (cam_sync_state.optional_sync_mode_case() == DeviceSerialization::CameraSynchronizationState::kSyncMode) {
            switch (cam_sync_state.sync_mode()) {
            case DeviceSerialization::CameraSynchronizationState::MASTER:
                module->set_mode_master();
                break;
            case DeviceSerialization::CameraSynchronizationState::SLAVE:
                module->set_mode_slave();
                break;
            case DeviceSerialization::CameraSynchronizationState::STANDALONE:
                module->set_mode_standalone();
                break;
            default:
                break;
            }
        }
    }

    void operator()(I_DigitalCrop *module) {
        if (!module) {
            return;
        }

        if (!state_.has_digital_crop_state()) {
            return;
        }

        const auto &digital_crop_state = state_.digital_crop_state();
        if (digital_crop_state.optional_enabled_case() == DeviceSerialization::DigitalCropState::kEnabled) {
            module->enable(digital_crop_state.enabled());
        }

        if (digital_crop_state.has_region()) {
            const auto &region = digital_crop_state.region();
            module->set_window_region(std::make_tuple(region.x1(), region.y1(), region.x2(), region.y2()));
        }
    }

    void operator()(I_DigitalEventMask *module) {
        if (!module) {
            return;
        }

        if (!state_.has_digital_event_mask_state()) {
            return;
        }

        const auto &digital_event_mask_state = state_.digital_event_mask_state();
        const auto mask_ptrs                 = module->get_pixel_masks();
        if (module->get_pixel_masks().size() < static_cast<size_t>(digital_event_mask_state.mask_size())) {
            MV_SDK_LOG_WARNING() << "Mismatched pixel masks size, some mask will not be loaded";
        }
        for (int i = 0; i < digital_event_mask_state.mask_size(); ++i) {
            if (static_cast<size_t>(i) < mask_ptrs.size()) {
                mask_ptrs[i]->set_mask(digital_event_mask_state.mask(i).x(), digital_event_mask_state.mask(i).y(),
                                       digital_event_mask_state.mask(i).enabled());
            }
        }
    }

    void operator()(I_ErcModule *module) {
        if (!module) {
            return;
        }

        if (!state_.has_event_rate_control_state()) {
            return;
        }

        const auto &erc_state = state_.event_rate_control_state();
        if (erc_state.optional_enabled_case() == DeviceSerialization::EventRateControlState::kEnabled) {
            module->enable(erc_state.enabled());
        }
        if (erc_state.optional_cd_event_count_case() == DeviceSerialization::EventRateControlState::kCdEventCount) {
            module->set_cd_event_count(erc_state.cd_event_count());
        }
    }

    void operator()(I_EventRateActivityFilterModule *module) {
        if (!module) {
            return;
        }

        const auto supported_thresholds = module->is_thresholds_supported();

        if (state_.has_event_rate_activity_filter_state()) {
            const auto &nfl_state = state_.event_rate_activity_filter_state();
            if (nfl_state.optional_enabled_case() == DeviceSerialization::EventRateActivityFilterState::kEnabled) {
                module->enable(nfl_state.enabled());
            }

            I_EventRateActivityFilterModule::thresholds thresholds = {0, 0, 0, 0};
            if (nfl_state.optional_lower_start_rate_threshold_case() ==
                DeviceSerialization::EventRateActivityFilterState::kLowerStartRateThreshold
                && supported_thresholds.lower_bound_start == 1) {
                thresholds.lower_bound_start = nfl_state.lower_start_rate_threshold();
            }
            if (nfl_state.optional_lower_stop_rate_threshold_case() ==
                DeviceSerialization::EventRateActivityFilterState::kLowerStopRateThreshold
                && supported_thresholds.lower_bound_stop == 1) {
                thresholds.lower_bound_stop = nfl_state.lower_stop_rate_threshold();
            }
            if (nfl_state.optional_upper_start_rate_threshold_case() ==
                DeviceSerialization::EventRateActivityFilterState::kUpperStartRateThreshold
                && supported_thresholds.upper_bound_start == 1) {
                thresholds.upper_bound_start = nfl_state.upper_start_rate_threshold();
            }
            if (nfl_state.optional_upper_stop_rate_threshold_case() ==
                DeviceSerialization::EventRateActivityFilterState::kUpperStopRateThreshold
                && supported_thresholds.upper_bound_stop == 1) {
                thresholds.upper_bound_stop = nfl_state.upper_stop_rate_threshold();
            }

            module->set_thresholds(thresholds);
            return;
        }

        // State serialized from metavision < 4.6.0
        if (!state_.has_event_rate_noise_filter_state()) {
            return;
        }

        const auto &nfl_state = state_.event_rate_noise_filter_state();
        if (nfl_state.optional_enabled_case() == DeviceSerialization::EventRateNoiseFilterState::kEnabled) {
            module->enable(nfl_state.enabled());
        }

        if (nfl_state.optional_event_rate_threshold_case() ==
            DeviceSerialization::EventRateNoiseFilterState::kEventRateThreshold) {
            const auto th = nfl_state.event_rate_threshold();

            I_EventRateActivityFilterModule::thresholds thresholds = {0, 0, 0, 0};
            // Find best candidate for threshold
            if (supported_thresholds.lower_bound_start == 1) {
                thresholds.lower_bound_start = th;
            } else if (supported_thresholds.upper_bound_start == 1) {
                thresholds.upper_bound_start = th;
            } else if (supported_thresholds.lower_bound_stop == 1) {
                thresholds.lower_bound_stop = th;
            } else if (supported_thresholds.upper_bound_stop == 1) {
                thresholds.upper_bound_stop = th;
            }
            module->set_thresholds(thresholds);
        }
    }

    void operator()(I_EventTrailFilterModule *module) {
        if (!module) {
            return;
        }

        if (!state_.has_event_trail_filter_state()) {
            return;
        }

        const auto &etf_state = state_.event_trail_filter_state();
        if (etf_state.optional_enabled_case() == DeviceSerialization::EventTrailFilterState::kEnabled) {
            module->enable(etf_state.enabled());
        }
        if (etf_state.optional_filtering_type_case() == DeviceSerialization::EventTrailFilterState::kFilteringType) {
            switch (etf_state.filtering_type()) {
            case DeviceSerialization::EventTrailFilterState::TRAIL:
                module->set_type(I_EventTrailFilterModule::Type::TRAIL);
                break;
            case DeviceSerialization::EventTrailFilterState::STC_CUT_TRAIL:
                module->set_type(I_EventTrailFilterModule::Type::STC_CUT_TRAIL);
                break;
            case DeviceSerialization::EventTrailFilterState::STC_KEEP_TRAIL:
                module->set_type(I_EventTrailFilterModule::Type::STC_KEEP_TRAIL);
                break;
            default:
                break;
            }
        }
        if (etf_state.optional_threshold_case() == DeviceSerialization::EventTrailFilterState::kThreshold) {
            module->set_threshold(etf_state.threshold());
        }
    }

    void operator()(I_HW_Register *module) {
        if (!module) {
            return;
        }

        if (!state_.has_hw_register_state()) {
            return;
        }

        const auto &hw_register_state = state_.hw_register_state();
        for (int i = 0; i < hw_register_state.num_access_size(); ++i) {
            const auto &access = hw_register_state.num_access(i);
            module->write_register(access.address(), access.value());
        }
        for (int i = 0; i < hw_register_state.str_access_size(); ++i) {
            const auto &access = hw_register_state.str_access(i);
            module->write_register(access.address(), access.value());
        }
        for (int i = 0; i < hw_register_state.bitfield_access_size(); ++i) {
            const auto &access = hw_register_state.bitfield_access(i);
            module->write_register(access.address(), access.bitfield(), access.value());
        }
    }

    void operator()(I_LL_Biases *module) {
        if (!module) {
            return;
        }

        if (!state_.has_ll_biases_state()) {
            return;
        }

        const auto &ll_biases_state = state_.ll_biases_state();
        for (int i = 0; i < ll_biases_state.bias_size(); ++i) {
            const auto &bias_state = ll_biases_state.bias(i);
            try {
                if (module->get(bias_state.name()) != bias_state.value()) {
                    module->set(bias_state.name(), bias_state.value());
                }
            } catch (HalException &e) {
                switch (e.code().value()) {
                case HalErrorCode::NonExistingValue:
                    MV_SDK_LOG_WARNING() << "Ignored unavailable bias:" << bias_state.name();
                    break;
                case HalErrorCode::OperationNotPermitted:
                    MV_SDK_LOG_WARNING() << "Ignored read-only bias:" << bias_state.name();
                    break;
                case HalErrorCode::ValueOutOfRange:
                    MV_SDK_LOG_WARNING() << "Ignored out of range bias:" << bias_state.name()
                                         << "with value:" << bias_state.value();
                    break;
                }
            }
        }
    }

    void operator()(I_ROI *module) {
        if (!module) {
            return;
        }

        if (!state_.has_roi_state()) {
            return;
        }

        const auto &roi_state = state_.roi_state();
        if (roi_state.optional_mode_case() == DeviceSerialization::RegionOfInterestState::kMode) {
            module->set_mode(roi_state.mode() == DeviceSerialization::RegionOfInterestState::ROI ? I_ROI::Mode::ROI :
                                                                                                   I_ROI::Mode::RONI);
        }

        std::vector<I_ROI::Window> windows;
        if (roi_state.window_size() > 0) {
            for (int i = 0; i < roi_state.window_size(); ++i) {
                windows.emplace_back(roi_state.window(i).x(), roi_state.window(i).y(), roi_state.window(i).width(),
                                     roi_state.window(i).height());
            }
            module->set_windows(windows);
        }

        if (roi_state.columns_size() > 0 && roi_state.rows_size() > 0) {
            std::vector<bool> cols;
            for (int i = 0; i < roi_state.columns_size(); ++i) {
                cols.push_back(roi_state.columns(i));
            }

            std::vector<bool> rows;
            for (int i = 0; i < roi_state.rows_size(); ++i) {
                rows.push_back(roi_state.rows(i));
            }
            module->set_lines(cols, rows);
        }

        if (roi_state.optional_enabled_case() == DeviceSerialization::RegionOfInterestState::kEnabled) {
            module->enable(roi_state.enabled());
        }
    }

    void operator()(I_TriggerIn *module) {
        if (!module) {
            return;
        }

        if (!state_.has_trigger_in_state()) {
            return;
        }

        const auto &trigger_in_state = state_.trigger_in_state();
        for (int i = 0; i < trigger_in_state.channel_status_size(); ++i) {
            const auto &channel_status = trigger_in_state.channel_status(i);
            I_TriggerIn::Channel channel;
            switch (channel_status.channel()) {
            case DeviceSerialization::TriggerInState::MAIN:
                channel = I_TriggerIn::Channel::Main;
                break;
            case DeviceSerialization::TriggerInState::AUX:
                channel = I_TriggerIn::Channel::Aux;
                break;
            case DeviceSerialization::TriggerInState::LOOPBACK:
                channel = I_TriggerIn::Channel::Loopback;
                break;
            default:
                break;
            }
            if (channel_status.enabled()) {
                module->enable(channel);
            } else {
                module->disable(channel);
            }
        }
    }

    void operator()(I_TriggerOut *module) {
        if (!module) {
            return;
        }

        if (!state_.has_trigger_out_state()) {
            return;
        }

        const auto &trigger_out_state = state_.trigger_out_state();
        if (trigger_out_state.optional_enabled_case() == DeviceSerialization::TriggerOutState::kEnabled) {
            if (trigger_out_state.enabled()) {
                module->enable();
            } else {
                module->disable();
            }
        }
        if (trigger_out_state.optional_period_case() == DeviceSerialization::TriggerOutState::kPeriod) {
            module->set_period(trigger_out_state.period());
        }
        if (trigger_out_state.optional_duty_cycle_case() == DeviceSerialization::TriggerOutState::kDutyCycle) {
            module->set_duty_cycle(trigger_out_state.duty_cycle());
        }
    }

private:
    const DeviceSerialization::DeviceState &state_;
};

std::ostream &save_device(const Device &d, std::ostream &os) {
    DeviceSerialization::DeviceState state;

    FacilitySerializer serializer(state);
    for (auto facility : get_serializable_facilities(d)) {
        std::visit(serializer, facility);
    }

    google::protobuf::util::JsonPrintOptions options;
    options.add_whitespace                = true;
    #if (GOOGLE_PROTOBUF_VERSION >= 5026000)
    options.always_print_fields_with_no_presence = true;
    #else
    options.always_print_primitive_fields = true;
    #endif
    options.preserve_proto_field_names    = true;

    std::string json;
    google::protobuf::util::MessageToJsonString(state, &json, options);
    os << json;

    return os;
}

std::istream &load_device(Device &d, std::istream &is) {
    std::stringstream sstream;
    sstream << is.rdbuf();

    const auto str = sstream.str();
    if (!str.empty()) {
        google::protobuf::util::JsonParseOptions options;
        DeviceSerialization::DeviceState state;

        options.ignore_unknown_fields = true;

        auto parse_status = google::protobuf::util::JsonStringToMessage(str, &state, options);
        if (!parse_status.ok()) {
            throw CameraException(CameraErrorCode::InvalidArgument,
                                  "Failed to parse camera settings: " + parse_status.ToString());
        }

        FacilityDeserializer deserializer(state);
        for (auto facility : get_serializable_facilities(d)) {
            std::visit(deserializer, facility);
        }
    }

    return is;
}

#else // !HAS_PROTOBUF

std::ostream &save_device(const Device &d, std::ostream &os) {
    return os;
}

std::istream &load_device(Device &d, std::istream &is) {
    return is;
}

#endif // HAS_PROTOBUF

} // namespace Metavision
