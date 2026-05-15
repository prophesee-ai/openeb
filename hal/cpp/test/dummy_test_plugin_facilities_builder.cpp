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

#include <cstdint>
#include <string>
#include <map>

#include "metavision/hal/decoders/evt3/evt3_decoder.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/facilities/i_antiflicker_module.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/facilities/i_digital_crop.h"
#include "metavision/hal/facilities/i_digital_event_mask.h"
#include "metavision/hal/facilities/i_erc_module.h"
#include "metavision/hal/facilities/i_event_rate_activity_filter_module.h"
#include "metavision/hal/facilities/i_event_trail_filter_module.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "metavision/hal/facilities/i_ll_biases.h"
#include "metavision/hal/facilities/i_monitoring.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/hal/facilities/i_roi.h"
#include "metavision/hal/facilities/i_trigger_in.h"
#include "metavision/hal/facilities/i_trigger_out.h"
#include "metavision/hal/plugin/plugin.h"
#include "metavision/hal/plugin/plugin_entrypoint.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/utils/file_discovery.h"
#include "metavision/hal/utils/hal_software_info.h"
#include <metavision/hal/utils/camera_discovery.h>

#include "dummy_test_plugin_facilities.h"
#include "dummy_raw_data_producer.h"

using namespace Metavision;

namespace {

struct DummyROI : public I_ROI {
    bool enable(bool state) override {
        enabled_ = state;
        return true;
    }

    bool is_enabled() const override {
        return enabled_;
    }

    bool set_mode(const Mode &mode) override {
        mode_ = mode;
        return true;
    }

    Mode get_mode() const override {
        return mode_;
    }

    size_t get_max_supported_windows_count() const override {
        return 5;
    }

    bool set_lines(const std::vector<bool> &cols, const std::vector<bool> &rows) override {
        rows_ = rows;
        cols_ = cols;
        return true;
    }

    bool get_lines(std::vector<bool> &cols, std::vector<bool> &rows) const override {
        cols = cols_;
        rows = rows_;
        return true;
    }

    bool set_windows_impl(const std::vector<Window> &windows) override {
        windows_ = windows;
        return true;
    }

    std::vector<Window> get_windows() const override {
        return windows_;
    }

    bool enabled_{false};
    Mode mode_;
    std::vector<Window> windows_;
    std::vector<bool> rows_, cols_;
};

struct DummyFileHWIdentification : public I_HW_Identification {
    DummyFileHWIdentification(const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info,
                              const RawFileHeader &header) :
        I_HW_Identification(plugin_sw_info), header_(header) {}

    std::string get_serial() const {
        return std::string();
    }

    SensorInfo get_sensor_info() const {
        return SensorInfo({0, 0, "Gen0.0"});
    }

    std::vector<std::string> get_available_data_encoding_formats() const {
        return std::vector<std::string>();
    }

    std::string get_current_data_encoding_format() const {
        return std::string();
    }

    std::string get_integrator() const {
        return header_.get_camera_integrator_name();
    }

    std::string get_connection_type() const {
        return std::string();
    }

    DeviceConfigOptionMap get_device_config_options_impl() const {
        return {};
    }

    RawFileHeader header_;
};

struct DummyHWRegister : public I_HW_Register {
    void write_register(uint32_t address, uint32_t v) override {
        hex_accesses_[address] = v;
    }

    void write_register(const std::string &address, uint32_t v) override {
        str_accesses_[address] = v;
    }

    uint32_t read_register(uint32_t address) override {
        return hex_accesses_[address];
    }

    uint32_t read_register(const std::string &address) override {
        return str_accesses_[address];
    }

    void write_register(const std::string &address, const std::string &bitfield, uint32_t v) override {
        bitfield_accesses_[std::make_pair(address, bitfield)] = v;
    }

    uint32_t read_register(const std::string &address, const std::string &bitfield) override {
        return bitfield_accesses_[std::make_pair(address, bitfield)];
    }

    std::map<uint32_t, uint32_t> hex_accesses_;
    std::map<std::string, uint32_t> str_accesses_;
    std::map<std::pair<std::string, std::string>, uint32_t> bitfield_accesses_;
};

struct DummyLLBiases : public I_LL_Biases {
    DummyLLBiases(const DeviceConfig &device_config) :
        I_LL_Biases(device_config),
        biases_{
            {"dummy", std::make_pair(LL_Bias_Info(-10, 10, "dummy desc", true, "dummy category"), 1)},
            {"a", std::make_pair(LL_Bias_Info(-10, 10, "", true), 0)},
            {"b", std::make_pair(LL_Bias_Info(-10, 10, "", false), 0)},
            {"c", std::make_pair(LL_Bias_Info(-10, 10, "", true), 0)},
        } {}

    std::map<std::string, int> get_all_biases() const override {
        std::map<std::string, int> m;
        for (const auto &bias : biases_) {
            m[bias.first] = bias.second.second;
        }
        return m;
    }

    bool set_impl(const std::string &bias_name, int bias_value) override {
        biases_[bias_name].second = bias_value;
        return true;
    }

    int get_impl(const std::string &bias_name) const override {
        return biases_.find(bias_name)->second.second;
    }

    bool get_bias_info_impl(const std::string &bias_name, LL_Bias_Info &bias_info) const override {
        auto it = biases_.find(bias_name);
        if (it == biases_.end())
            return false;
        bias_info = it->second.first;
        return true;
    }

    std::map<std::string, std::pair<LL_Bias_Info, int>> biases_;
};

struct DummyFileDiscovery : public FileDiscovery {
    bool discover(DeviceBuilder &device_builder, std::unique_ptr<std::istream> &file, const RawFileHeader &header,
                  const RawFileConfig &file_config) override {
        device_builder.add_facility(
            std::make_unique<DummyFileHWIdentification>(device_builder.get_plugin_software_info(), header));

        return true;
    }
};

struct DummyDigitalEvenMask : public I_DigitalEventMask {
    class DummyPixelMask : public I_PixelMask {
        uint32_t x_{0}, y_{0};
        bool enabled_{false};

        bool set_mask(uint32_t x, uint32_t y, bool enabled) override final {
            x_       = x;
            y_       = y;
            enabled_ = enabled;
            return true;
        }
        std::tuple<uint32_t, uint32_t, bool> get_mask() const override final {
            return std::make_tuple(x_, y_, enabled_);
        }
    };

    DummyDigitalEvenMask() {
        pixel_masks_.push_back(std::make_shared<DummyPixelMask>());
        pixel_masks_.push_back(std::make_shared<DummyPixelMask>());
        pixel_masks_.push_back(std::make_shared<DummyPixelMask>());
    }
    const std::vector<I_PixelMaskPtr> &get_pixel_masks() const override final {
        return pixel_masks_;
    }

private:
    std::vector<I_PixelMaskPtr> pixel_masks_;
};

struct DummyMonitoring : public I_Monitoring {
    int get_temperature() override final {
        return 12;
    }
    int get_illumination() override final {
        return 34;
    }
    int get_pixel_dead_time() override final {
        return 56;
    }
};

class DummyDigitalCrop : public I_DigitalCrop {
private:
    bool enabled_ = false;
    Region region_;

public:
    bool enable(bool state) override {
        enabled_ = state;
        return true;
    }
    bool is_enabled() const override {
        return enabled_;
    }
    bool set_window_region(const Region &region, bool reset_origin) override {
        using std::get;
        if (get<0>(region) > get<2>(region)) {
            throw std::runtime_error("Crop region error");
        }
        if (get<1>(region) > get<3>(region)) {
            throw std::runtime_error("Crop region error");
        }
        region_ = region;
        return true;
    }
    Region get_window_region() const override {
        return region_;
    }
};

class DummyTriggerIn : public I_TriggerIn {
private:
    std::map<Channel, short> channel_map_{{Channel::Main, 0}, {Channel::Aux, 1}, {Channel::Loopback, 2}};
    std::map<Channel, bool> status_map_{{Channel::Main, false}, {Channel::Aux, false}, {Channel::Loopback, false}};

public:
    bool enable(const Channel &channel) override {
        if (status_map_.find(channel) == status_map_.end()) {
            return false;
        }
        status_map_[channel] = true;
        return true;
    }

    bool disable(const Channel &channel) override {
        if (status_map_.find(channel) == status_map_.end()) {
            return false;
        }
        status_map_[channel] = false;
        return true;
    }

    bool is_enabled(const Channel &channel) const override {
        auto it = status_map_.find(channel);
        if (it == status_map_.end()) {
            return false;
        }
        return it->second;
    }

    std::map<Channel, short> get_available_channels() const override {
        return channel_map_;
    }
};

class DummyTriggerOut : public I_TriggerOut {
public:
    uint32_t get_period() const override {
        return period_;
    }

    bool set_period(uint32_t period_us) override {
        period_ = period_us;
        return true;
    }

    double get_duty_cycle() const override {
        return duty_cycle_;
    }

    bool set_duty_cycle(double period_ratio) override {
        duty_cycle_ = period_ratio;
        return true;
    }

    bool enable() override {
        enabled_ = true;
        return true;
    }

    bool disable() override {
        enabled_ = false;
        return true;
    }

    bool is_enabled() const override {
        return enabled_;
    }

private:
    bool enabled_{false};
    uint32_t period_{1000};
    double duty_cycle_{0.1};
};

class DummyTrailFilterModule : public Metavision::I_EventTrailFilterModule {
public:
    virtual std::set<Type> get_available_types() const override {
        return {Type::STC_KEEP_TRAIL, Type::STC_CUT_TRAIL, Type::TRAIL};
    }

    virtual bool enable(bool state) override {
        enabled_ = state;
        return true;
    }

    virtual bool is_enabled() const override {
        return enabled_;
    }

    virtual bool set_type(Type type) override {
        type_ = type;
        return true;
    }

    virtual Type get_type() const override {
        return type_;
    }

    virtual bool set_threshold(uint32_t threshold) override {
        threshold_ = threshold;
        return true;
    }

    virtual uint32_t get_threshold() const override {
        return threshold_;
    }

    virtual uint32_t get_min_supported_threshold() const override {
        return 0;
    }

    virtual uint32_t get_max_supported_threshold() const override {
        return 1000;
    }

private:
    uint32_t threshold_{1};
    Type type_{Type::TRAIL};
    bool enabled_{false};
};

class DummyCameraSynchronization : public I_CameraSynchronization {
public:
    virtual bool set_mode_standalone() override {
        mode_ = I_CameraSynchronization::SyncMode::STANDALONE;
        return true;
    }

    virtual bool set_mode_master() override {
        mode_ = I_CameraSynchronization::SyncMode::MASTER;
        return true;
    }

    virtual bool set_mode_slave() override {
        mode_ = I_CameraSynchronization::SyncMode::SLAVE;
        return true;
    }

    virtual SyncMode get_mode() const override {
        return mode_;
    }

private:
    I_CameraSynchronization::SyncMode mode_ = I_CameraSynchronization::SyncMode::STANDALONE;
};

class DummyGeometry : public I_Geometry {
public:
    DummyGeometry(int width, int height) : width_(width), height_(height) {}

    virtual int get_width() const override {
        return width_;
    }

    virtual int get_height() const override {
        return height_;
    }

private:
    int width_;
    int height_;
};

class DummyHWIdentification : public I_HW_Identification {
public:
    DummyHWIdentification(const std::shared_ptr<I_PluginSoftwareInfo> &plugin_info) :
        I_HW_Identification(plugin_info) {}

    virtual std::string get_serial() const override {
        return "";
    }

    virtual SensorInfo get_sensor_info() const override {
        return SensorInfo();
    }

    virtual std::vector<std::string> get_available_data_encoding_formats() const override {
        return {"EVT3"};
    }

    virtual std::string get_current_data_encoding_format() const override {
        return "EVT3";
    }

    virtual std::string get_integrator() const override {
        return "";
    }

    virtual SystemInfo get_system_info() const override {
        return SystemInfo();
    }

    virtual std::string get_connection_type() const override {
        return "";
    }

    virtual DeviceConfigOptionMap get_device_config_options_impl() const override {
        return DeviceConfigOptionMap();
    }
};

class DummyAntiFlickerModule : public I_AntiFlickerModule {
public:
    virtual bool enable(bool b) override {
        enabled_ = b;
        return true;
    }

    virtual bool is_enabled() const override {
        return enabled_;
    }

    virtual bool set_frequency_band(uint32_t low_freq, uint32_t high_freq) override {
        low_freq_  = low_freq;
        high_freq_ = high_freq;
        return true;
    }

    virtual uint32_t get_band_low_frequency() const override {
        return low_freq_;
    }

    virtual uint32_t get_band_high_frequency() const override {
        return high_freq_;
    }

    virtual uint32_t get_min_supported_frequency() const override {
        return 0;
    }

    virtual uint32_t get_max_supported_frequency() const override {
        return 1000000;
    };

    virtual bool set_filtering_mode(I_AntiFlickerModule::AntiFlickerMode mode) override {
        mode_ = mode;
        return true;
    }

    virtual AntiFlickerMode get_filtering_mode() const override {
        return mode_;
    }

    virtual bool set_duty_cycle(float duty_cycle) override {
        duty_cycle_ = duty_cycle;
        return true;
    }

    virtual float get_duty_cycle() const override {
        return duty_cycle_;
    }

    virtual float get_min_supported_duty_cycle() const override {
        return 0.0;
    }

    virtual float get_max_supported_duty_cycle() const override {
        return 100.0;
    }

    virtual bool set_start_threshold(uint32_t threshold) override {
        start_threshold_ = threshold;
        return true;
    }

    virtual bool set_stop_threshold(uint32_t threshold) override {
        stop_threshold_ = threshold;
        return true;
    }

    virtual uint32_t get_start_threshold() const override {
        return start_threshold_;
    }

    virtual uint32_t get_stop_threshold() const override {
        return stop_threshold_;
    }

    virtual uint32_t get_min_supported_start_threshold() const override {
        return 0;
    }

    virtual uint32_t get_max_supported_start_threshold() const override {
        return 100000;
    }

    virtual uint32_t get_min_supported_stop_threshold() const override {
        return 0;
    }

    virtual uint32_t get_max_supported_stop_threshold() const override {
        return 100000;
    }

private:
    bool enabled_                              = false;
    I_AntiFlickerModule::AntiFlickerMode mode_ = I_AntiFlickerModule::AntiFlickerMode::BAND_PASS;
    uint32_t low_freq_                         = 0;
    uint32_t high_freq_                        = 100;
    uint32_t start_threshold_                  = 0;
    uint32_t stop_threshold_                   = 0;
    float duty_cycle_                          = 50.0;
};

struct DummyErcModule : public I_ErcModule {
public:
    virtual bool enable(bool b) override {
        enabled_ = b;
        return true;
    }

    virtual bool is_enabled() const override {
        return enabled_;
    }

    virtual uint32_t get_count_period() const override {
        return count_period_;
    }

    virtual bool set_cd_event_count(uint32_t event_count) override {
        cd_event_count_ = event_count;
        return true;
    }

    virtual uint32_t get_min_supported_cd_event_count() const override {
        return 2 * count_period_;
    }

    virtual uint32_t get_max_supported_cd_event_count() const override {
        return 874 * count_period_;
    }

    virtual uint32_t get_cd_event_count() const override {
        return cd_event_count_;
    }

    bool set_cd_event_rate(uint32_t events_per_sec) override {
        uint32_t count_period = get_count_period();
        set_cd_event_count(static_cast<uint32_t>(static_cast<uint64_t>(events_per_sec) * count_period / 1000000));
        return true;
    }

    virtual void erc_from_file(const std::string &) override {}

private:
    bool enabled_            = false;
    uint32_t count_period_   = 1000;
    uint32_t cd_event_count_ = 1000;
};

struct DummyNFLModule : public I_EventRateActivityFilterModule {
public:
    virtual bool enable(bool b) override {
        enabled_ = b;
        return true;
    }

    virtual bool is_enabled() const override {
        return enabled_;
    }

    virtual thresholds is_thresholds_supported() const override {
        return {1, 1, 1, 1};
    }

    virtual bool set_thresholds(const thresholds &thresholds_ev_s) override {
        threshold_  = thresholds_ev_s.lower_bound_start;
        thresholds_ = thresholds_ev_s;
        return true;
    }

    virtual thresholds get_thresholds() const override {
        return thresholds_;
    }

    virtual thresholds get_min_supported_thresholds() const override {
        return {0, 0, 0, 0};
    }

    virtual thresholds get_max_supported_thresholds() const override {
        return {200000, 200000, 1600000000, 1600000000};
    }

private:
    bool enabled_          = false;
    uint32_t threshold_    = 1000;
    thresholds thresholds_ = {300'000, 2'800'000, 500'000'000, 800'000'000};
};

struct DummyCameraDiscovery : public CameraDiscovery {
    SerialList list() override final {
        return SerialList{"__DummyTest__"};
    }
    SystemList list_available_sources() override final {
        return SystemList{PluginCameraDescription{"__DummyTest__", ConnectionType::PROPRIETARY_LINK}};
    }
    bool discover(DeviceBuilder &device_builder, const std::string &serial, const DeviceConfig &config) override final {
        device_builder.add_facility(std::make_unique<DummyDigitalCrop>());
        device_builder.add_facility(std::make_unique<DummyDigitalEvenMask>());
        device_builder.add_facility(std::make_unique<DummyMonitoring>());
        device_builder.add_facility(std::make_unique<DummyFacilityV3>());
        device_builder.add_facility(std::make_unique<DummyTriggerIn>());
        device_builder.add_facility(std::make_unique<DummyTriggerOut>());
        device_builder.add_facility(std::make_unique<DummyLLBiases>(config));
        device_builder.add_facility(std::make_unique<DummyTrailFilterModule>());
        device_builder.add_facility(std::make_unique<DummyAntiFlickerModule>());
        device_builder.add_facility(std::make_unique<DummyErcModule>());
        device_builder.add_facility(std::make_unique<DummyNFLModule>());
        device_builder.add_facility(std::make_unique<DummyHWRegister>());
        device_builder.add_facility(std::make_unique<DummyROI>());
        // To make dummy plugin usable in Metavision::Camera
        device_builder.add_facility(std::make_unique<DummyHWIdentification>(device_builder.get_plugin_software_info()));
        device_builder.add_facility(std::make_unique<DummyGeometry>(640, 480));
        device_builder.add_facility(std::make_unique<I_EventsStream>(
            std::make_unique<DummyRawDataProducer>(),
            std::make_unique<DummyHWIdentification>(device_builder.get_plugin_software_info())));
        device_builder.add_facility(make_evt3_decoder(false, 640, 480));
        device_builder.add_facility(std::make_unique<DummyCameraSynchronization>());
        return true;
    }

    bool is_for_local_camera() const override final {
        return true;
    }
};

} // namespace

void initialize_plugin(void *plugin_ptr) {
    Metavision::Plugin &plugin = Metavision::plugin_cast(plugin_ptr);
    plugin.set_integrator_name("__DummyTestPlugin__");
    plugin.set_plugin_info(Metavision::get_hal_software_info());
    plugin.set_hal_info(Metavision::get_hal_software_info());

    plugin.add_file_discovery(std::make_unique<DummyFileDiscovery>());
    plugin.add_camera_discovery(std::make_unique<DummyCameraDiscovery>());
}
