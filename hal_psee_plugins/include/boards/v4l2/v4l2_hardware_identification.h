#include "metavision/hal/facilities/i_hw_identification.h"
#include "boards/v4l2/v4l2_device.h"

#include <linux/videodev2.h>

namespace Metavision {
using V4l2Capability     = struct v4l2_capability;

class V4l2HwIdentification : public I_HW_Identification {
    std::shared_ptr<V4L2DeviceControl> ctrl_;

public:
    V4l2HwIdentification(std::shared_ptr<V4L2DeviceControl> ctrl, const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info);

    virtual SensorInfo get_sensor_info() const override;
    virtual std::vector<std::string> get_available_data_encoding_formats() const override;
    virtual std::string get_current_data_encoding_format() const override;
    virtual std::string get_serial() const override;
    virtual std::string get_integrator() const override;
    virtual std::string get_connection_type() const override;

protected:
    virtual DeviceConfigOptionMap get_device_config_options_impl() const override;
};
}
