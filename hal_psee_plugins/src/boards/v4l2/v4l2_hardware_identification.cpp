#include "boards/v4l2/v4l2_hardware_identification.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"

namespace Metavision {

V4l2HwIdentification::V4l2HwIdentification(const V4l2Capability cap,
                                           const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info,
                                           const SensorDescriptor &sensor_descriptor) :
    I_HW_Identification(plugin_sw_info), cap_(cap), sensor_descriptor_(sensor_descriptor) {}

long V4l2HwIdentification::get_system_id() const {
    // @TODO Retrieve those info through V4L2
    return 1234;
}
I_HW_Identification::SensorInfo V4l2HwIdentification::get_sensor_info() const {
    // @TODO Retrieve those info through V4L2
    return sensor_descriptor_.info;
}
std::vector<std::string> V4l2HwIdentification::get_available_data_encoding_formats() const {
    // @TODO Retrieve those info through V4L2
    auto format = get_current_data_encoding_format();
    auto pos = format.find(";");
    if (pos != std::string::npos) {
        auto evt_type = format.substr(0, pos);
        return {evt_type};
    }
    return {};
}
std::string V4l2HwIdentification::get_current_data_encoding_format() const {
    // @TODO Retrieve those info through V4L2
    return sensor_descriptor_.encoding_format;
}
std::string V4l2HwIdentification::get_serial() const {
    std::stringstream ss;
    ss << cap_.card;
    return ss.str();
}
std::string V4l2HwIdentification::get_integrator() const {
    std::stringstream ss;
    ss << cap_.driver;
    return ss.str();
}
std::string V4l2HwIdentification::get_connection_type() const {
    std::stringstream ss;
    ss << cap_.bus_info;
    return ss.str();
}
DeviceConfigOptionMap V4l2HwIdentification::get_device_config_options_impl() const {
    return {};
}
}
