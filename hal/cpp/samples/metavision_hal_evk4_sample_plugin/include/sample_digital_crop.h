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

#ifndef METAVISION_HAL_SAMPLE_DIGITAL_CROP_H
#define METAVISION_HAL_SAMPLE_DIGITAL_CROP_H

#include <memory>

#include <metavision/hal/facilities/i_digital_crop.h>

class SampleUSBConnection;

/// @brief Digital Crop sample implementation
/// All pixels outside of the cropping region will be dropped by the sensor
///
/// This class is the implementation of HAL's facility @ref Metavision::I_DigitalCrop
class SampleDigitalCrop : public Metavision::I_DigitalCrop {

public:
    SampleDigitalCrop(std::shared_ptr<SampleUSBConnection> usb_connection);
    bool enable(bool state) override;
    bool is_enabled() const override;
    bool set_window_region(const Region &region, bool reset_origin) override;
    Region get_window_region() const override;
private:
    std::shared_ptr<SampleUSBConnection> usb_connection_;
};


#endif // METAVISION_HAL_SAMPLE_DIGITAL_CROP_H
