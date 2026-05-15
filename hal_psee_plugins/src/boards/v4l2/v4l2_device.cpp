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
#include <cstring>
#include <filesystem>
#include "boards/v4l2/v4l2_device.h"

// Linux specific headers
#include <fcntl.h>
#include <linux/videodev2.h>
#include <string>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

#include "metavision/psee_hw_layer/boards/v4l2/v4l2_controls.h"

using namespace Metavision;

void Metavision::raise_error(const std::string &str) {
    throw std::runtime_error(str + " (" + std::to_string(errno) + " - " + std::strerror(errno) + ")");
}

V4L2DeviceControl::V4L2DeviceControl(const std::string &devpath) {
    struct stat st;
    if (-1 == stat(devpath.c_str(), &st))
        raise_error(devpath + "Cannot identify device.");

    if (!S_ISCHR(st.st_mode))
        throw std::runtime_error(devpath + " is not a device");

    media_fd_ = open(devpath.c_str(), O_RDWR | O_NONBLOCK, 0);
    if (-1 == media_fd_) {
        raise_error(devpath + "Cannot open media device");
    }

    enumerate_entities();

    auto video_ent = get_video_entity();
    if (video_ent == nullptr) {
        throw std::runtime_error("Could not find a v4l2 video device");
    }

    if (ioctl(video_ent->fd, VIDIOC_QUERYCAP, &cap_)) {
        if (EINVAL == errno) {
            throw std::runtime_error(devpath + " is not a V4L2 device");
        } else {
            raise_error("VIDIOC_QUERYCAP failed");
        }
    }

    if (!(cap_.capabilities & V4L2_CAP_VIDEO_CAPTURE))
        throw std::runtime_error(devpath + " is not video capture device");

    if (!(cap_.capabilities & V4L2_CAP_STREAMING))
        throw std::runtime_error(devpath + " does not support streaming i/o");

    auto sensor_ent = get_sensor_entity();
    if (sensor_ent == nullptr) {
        throw std::runtime_error("Could not find a v4l2 sensor subdevice");
    }

    // only expose sensor controls for now
    controls_ = std::make_shared<V4L2Controls>(sensor_ent->fd);
    // Note: this code expects the V4L2 device to be configured to output a supported format
}

V4l2Capability V4L2DeviceControl::get_capability() const {
    return cap_;
}

void V4L2DeviceControl::start() {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(get_video_entity()->fd, VIDIOC_STREAMON, &type))
        raise_error("VIDIOC_STREAMON failed");
}
void V4L2DeviceControl::stop() {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(get_video_entity()->fd, VIDIOC_STREAMOFF, &type))
        raise_error("VIDIOC_STREAMOFF failed");
}
void V4L2DeviceControl::reset() {}

std::shared_ptr<V4L2Controls> V4L2DeviceControl::get_controls() {
    return controls_;
}

int V4L2DeviceControl::enumerate_entities() {
    struct media_entity entity;
    int ret = 0;
    int id = 0;

    for (id = 0; ; id = entity.desc.id) {

        char target[1024];
        const std::filesystem::path sys_base = "/sys/dev/char/";

        memset(&entity.desc, 0, sizeof(entity.desc));
        entity.desc.id = id | MEDIA_ENT_ID_FLAG_NEXT;
        ret = ioctl(media_fd_, MEDIA_IOC_ENUM_ENTITIES, &entity.desc);
        if (ret < 0) {
            if (errno == EINVAL) {
                break;
            }
            MV_HAL_LOG_TRACE() << "MEDIA_IOC_ENUM_ENTITIES ioctl failed:" << strerror(errno);
            return -1;
        }

        MV_HAL_LOG_TRACE() << "Found entity: " << entity.desc.name;
        std::filesystem::path sys_path = sys_base / (std::to_string(entity.desc.v4l.major) + ":" + std::to_string(entity.desc.v4l.minor));

        ret = readlink(sys_path.c_str(), target, sizeof(target));
        if (ret < 0) {
            MV_HAL_LOG_TRACE() << "Could not readlink" << sys_path << strerror(errno);
            return -1;
        }
        target[ret] = '\0';

        std::filesystem::path dev_path(target);
        std::filesystem::path devpath = std::filesystem::path("/dev/") / dev_path.filename();

        entity.path = devpath;
        entity.type = entity.desc.type;
        entity.fd = open(devpath.c_str(), O_RDWR);

        entities_.push_back(entity);
    }
    return 0;
}

