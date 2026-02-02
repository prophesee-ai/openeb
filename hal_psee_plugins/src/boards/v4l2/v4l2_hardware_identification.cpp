#include "boards/v4l2/v4l2_hardware_identification.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "boards/v4l2/v4l2_device.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include <fcntl.h>

namespace Metavision {

V4l2HwIdentification::V4l2HwIdentification(std::shared_ptr<V4L2DeviceControl> ctrl,
                                           const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info) :
    I_HW_Identification(plugin_sw_info), ctrl_(ctrl) {}

I_HW_Identification::SensorInfo V4l2HwIdentification::get_sensor_info() const {
    auto controls = ctrl_->get_controls();
    auto sensor_ent = ctrl_->get_sensor_entity();
    std::string ent_name = std::string(sensor_ent->desc.name);

    if (ent_name.find("imx636") == 0) {
        return {4, 2, "IMX636"};
    } else if (ent_name.find("genx320") == 0) {
        return {320, 0, "GenX320"};
    } else {
        raise_error("Unknown sensor");
    }
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
    struct v4l2_format fmt {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    };

    if (ioctl(ctrl_->get_video_entity()->fd, VIDIOC_G_FMT, &fmt))
        raise_error("VIDIOC_G_FMT failed");

    switch (fmt.fmt.pix.pixelformat) {
    case v4l2_fourcc('P', 'S', 'E', 'E'): {
        StreamFormat format("EVT2");
        format["width"]                    = std::to_string(fmt.fmt.pix.width);
        format["height"]                   = std::to_string(fmt.fmt.pix.height);
        return format.to_string();
    }
    case v4l2_fourcc('P', 'S', 'E', '1'): {
        StreamFormat format("EVT21");
        format["endianness"]               = "legacy";
        format["width"]                    = std::to_string(fmt.fmt.pix.width);
        format["height"]                   = std::to_string(fmt.fmt.pix.height);
        return format.to_string();
    }
    case v4l2_fourcc('P', 'S', 'E', '2'): {
        StreamFormat format("EVT21");
        format["width"]                    = std::to_string(fmt.fmt.pix.width);
        format["height"]                   = std::to_string(fmt.fmt.pix.height);
        return format.to_string();
    }
    case v4l2_fourcc('P', 'S', 'E', '3'): {
        StreamFormat format("EVT3");
        format["width"]                    = std::to_string(fmt.fmt.pix.width);
        format["height"]                   = std::to_string(fmt.fmt.pix.height);
        return format.to_string();
    }
    default:
        throw std::runtime_error("Unsupported pixel format");
    }
}

std::string V4l2HwIdentification::get_serial() const {
    std::stringstream ss;
    ss << ctrl_->get_capability().card;
    return ss.str();
}
std::string V4l2HwIdentification::get_integrator() const {
    std::stringstream ss;
    ss << ctrl_->get_capability().driver;
    return ss.str();
}
std::string V4l2HwIdentification::get_connection_type() const {
    std::stringstream ss;
    ss << ctrl_->get_capability().bus_info;
    return ss.str();
}

DeviceConfigOptionMap V4l2HwIdentification::get_device_config_options_impl() const {
    return {};
}
}
