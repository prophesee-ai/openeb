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

#ifndef METAVISION_HAL_PSEE_PLUGINS_V4L2_DEVICE_H
#define METAVISION_HAL_PSEE_PLUGINS_V4L2_DEVICE_H

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <filesystem>
#include <linux/media.h>

#include <linux/videodev2.h>
#include <linux/v4l2-subdev.h>

#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/utils/device_control.h"
#include "metavision/psee_hw_layer/boards/v4l2/v4l2_controls.h"

namespace Metavision {

void raise_error(const std::string &str);

using V4l2Capability = struct v4l2_capability;

struct media_entity {
    int fd;
    std::filesystem::path path;
    uint32_t type;
    struct media_entity_desc desc;
};

class V4L2DeviceControl : public DeviceControl {
    V4l2Capability cap_;
    int media_fd_ = -1;
    std::vector<media_entity> entities_;
    std::shared_ptr<V4L2Controls> controls_;

public:
    /* Count the number of bytes received in the buffer. The complexity is log(n) */
    template<typename Data>
    static std::size_t nb_not_null_data(const Data *const buf_beg_addr, std::size_t length_in_bytes) {
        auto is_not_null = [](const auto &d) { return d != 0; };
        auto beg         = reinterpret_cast<const uint64_t *>(buf_beg_addr);
        auto end         = beg + length_in_bytes / sizeof(uint64_t);

        auto it_pp = std::partition_point(beg, end, is_not_null);
        return std::distance(beg, it_pp) * sizeof(*beg);
    }

    V4L2DeviceControl(const std::string &dev_name);
    virtual ~V4L2DeviceControl() = default;

    V4l2Capability get_capability() const;

    int get_media_fd() const {
        return media_fd_;
    }

    const struct media_entity *get_sensor_entity() const {
        auto sensor = std::find_if(entities_.begin(), entities_.end(),
                                   [](const auto &entity) { return entity.type == MEDIA_ENT_T_V4L2_SUBDEV_SENSOR; });

        if (sensor == entities_.end()) {
            return nullptr;
        }

        return &(*sensor);
    }

    const struct media_entity *get_video_entity() const {
        auto video = std::find_if(entities_.begin(), entities_.end(),
                                  [](const auto &entity) { return entity.type == MEDIA_ENT_T_DEVNODE_V4L; });

        if (video == entities_.end()) {
            return nullptr;
        }

        return &(*video);
    }

    bool can_crop(int fd) {
        struct v4l2_subdev_selection sel = {0};

        sel.which  = V4L2_SUBDEV_FORMAT_ACTIVE;
        sel.pad    = 0;
        sel.target = V4L2_SEL_TGT_CROP_ACTIVE;
        if (ioctl(fd, VIDIOC_SUBDEV_G_CROP, &sel) == -EINVAL) {
            MV_HAL_LOG_TRACE() << "device can't crop";
            return false;
        }
        return true;
    }

    void set_crop(int fd, const struct v4l2_rect &rect) {
        struct v4l2_subdev_selection sel = {0};

        sel.pad    = 0;
        sel.which  = V4L2_SUBDEV_FORMAT_ACTIVE;
        sel.target = V4L2_SEL_TGT_CROP;
        sel.r      = rect;
        if (ioctl(fd, VIDIOC_SUBDEV_S_SELECTION, &sel) < 0) {
            raise_error("VIDIOC_SUBDEV_S_SELECTION failed");
        }
    }

    void get_native_size(int fd, struct v4l2_rect &rect) {
        struct v4l2_subdev_selection sel = {0};

        sel.pad    = 0;
        sel.which  = V4L2_SUBDEV_FORMAT_ACTIVE;
        sel.target = V4L2_SEL_TGT_NATIVE_SIZE;
        if (ioctl(fd, VIDIOC_SUBDEV_G_SELECTION, &sel) < 0) {
            raise_error("VIDIOC_SUBDEV_G_SELECTION failed");
        }
        rect = sel.r;
    }

    void get_crop(int fd, struct v4l2_rect &rect) {
        struct v4l2_subdev_selection sel = {0};

        std::memset(&sel, 0, sizeof(sel));
        sel.pad    = 0;
        sel.which  = V4L2_SUBDEV_FORMAT_ACTIVE;
        sel.target = V4L2_SEL_TGT_CROP;
        if (ioctl(fd, VIDIOC_SUBDEV_G_SELECTION, &sel) < 0) {
            raise_error("VIDIOC_SUBDEV_G_SELECTION failed");
        }
        rect = sel.r;
    }

    int get_height() const {
        struct v4l2_format fmt{.type = V4L2_BUF_TYPE_VIDEO_CAPTURE};

        if (ioctl(get_video_entity()->fd, VIDIOC_G_FMT, &fmt))
            raise_error("VIDIOC_G_FMT failed");

        return fmt.fmt.pix.height;
    };

    int get_width() const {
        struct v4l2_format fmt{.type = V4L2_BUF_TYPE_VIDEO_CAPTURE};
        if (ioctl(get_video_entity()->fd, VIDIOC_G_FMT, &fmt))
            raise_error("VIDIOC_G_FMT failed");
        return fmt.fmt.pix.width;
    };

    int enumerate_entities();
    std::shared_ptr<V4L2Controls> get_controls();

    // DeviceControl
public:
    virtual void start() override;
    virtual void stop() override;
    virtual void reset() override;
};

class V4l2Synchronization : public I_CameraSynchronization {
public:
    virtual bool set_mode_standalone() override {
        return true;
    }
    virtual bool set_mode_master() override {
        return false;
    }
    virtual bool set_mode_slave() override {
        return false;
    }
    virtual SyncMode get_mode() const override {
        return SyncMode::STANDALONE;
    }
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINS_V4L2_DEVICE_H
