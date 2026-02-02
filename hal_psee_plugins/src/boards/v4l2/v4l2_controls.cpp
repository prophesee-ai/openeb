#include <cstring>
#include <optional>
#include <map>
#include <string>
#include <functional>
#include <cerrno>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/boards/v4l2/v4l2_controls.h"

namespace Metavision {
V4L2Controls::V4L2Control::V4L2Control(int fd) : fd_(fd) {
    memset(&ctrl, 0, sizeof(struct v4l2_ext_control));
}

int V4L2Controls::V4L2Control::push() {
    if (type() != V4L2_CTRL_TYPE_BUTTON) {
        throw std::runtime_error("Only button controls can be pushed");
    }

    return apply();
}
std::optional<int> V4L2Controls::V4L2Control::get_int(void) {
    if (query_.flags & V4L2_CTRL_FLAG_WRITE_ONLY) {
        return {};
    }

    switch(query_.type) {
        case V4L2_CTRL_TYPE_INTEGER:
            return std::optional<int>(static_cast<int>(ctrl.value));
        default:
            return {};
    }
}

std::optional<int64_t> V4L2Controls::V4L2Control::get_int64(void) {
    if (query_.flags & V4L2_CTRL_FLAG_WRITE_ONLY) {
        return {};
    }

    switch(query_.type) {
        case V4L2_CTRL_TYPE_INTEGER:
            return std::optional<int64_t>(static_cast<int64_t>(ctrl.value));
        case V4L2_CTRL_TYPE_INTEGER64:
            return std::optional<int64_t>(static_cast<int64_t>(ctrl.value64));
        default:
            return {};
    }
}

std::optional<bool> V4L2Controls::V4L2Control::get_bool(void) {
    if (query_.flags & V4L2_CTRL_FLAG_WRITE_ONLY) {
        return {};
    }

    switch(query_.type) {
        case V4L2_CTRL_TYPE_BOOLEAN:
            return std::optional<bool>(static_cast<bool>(ctrl.value));
        default:
            return {};
    }
}

std::optional<std::string> V4L2Controls::V4L2Control::get_str(void) {
    if (query_.flags & V4L2_CTRL_FLAG_WRITE_ONLY) {
        return {};
    }

    switch(query_.type) {
        case V4L2_CTRL_TYPE_STRING:
            return std::string(ctrl.string);
        default:
            return {};
    }
}

int V4L2Controls::V4L2Control::set_int(int value) {
    // guards
    if (query_.flags & V4L2_CTRL_FLAG_WRITE_ONLY)
        return -EPERM;

    if (query_.type != V4L2_CTRL_TYPE_INTEGER)
        return -EINVAL;

    // check within range
    if (value < query_.minimum || value > query_.maximum)
        return -EINVAL;

    ctrl.value = value;

    apply();
    return 0;
}

int V4L2Controls::V4L2Control::set_int64(std::int64_t value) {
    if (query_.flags & V4L2_CTRL_FLAG_WRITE_ONLY)
        return -EPERM;

    if (query_.type != V4L2_CTRL_TYPE_INTEGER64)
        return -EINVAL;

    // check within range
    if (value < query_.minimum || value > query_.maximum)
        return -EINVAL;

    ctrl.value64 = value;

    apply();
    return 0;
}

int V4L2Controls::V4L2Control::set_bool(bool value) {
    if (query_.flags & V4L2_CTRL_FLAG_WRITE_ONLY)
        return -EPERM;

    if (query_.type != V4L2_CTRL_TYPE_BOOLEAN)
        return -EINVAL;

    ctrl.value = value;
    return apply();

}

std::string V4L2Controls::V4L2Control::strtype() const {
    switch (type()) {
        case V4L2_CTRL_TYPE_INTEGER:
            return "integer";
        case V4L2_CTRL_TYPE_BOOLEAN:
            return "boolean";
        case V4L2_CTRL_TYPE_MENU:
            return "menu";
        case V4L2_CTRL_TYPE_BUTTON:
            return "button";
        case V4L2_CTRL_TYPE_INTEGER64:
            return "integer64";
        case V4L2_CTRL_TYPE_CTRL_CLASS:
            return "class";
        case V4L2_CTRL_TYPE_STRING:
            return "string";
        case V4L2_CTRL_TYPE_BITMASK:
            return "bitmask";
        case V4L2_CTRL_TYPE_INTEGER_MENU:
            return "integer_menu";
        default:
            return "unknown";
    }
}

bool V4L2Controls::V4L2Control::is_volatile() {
    return !!(query_.flags & V4L2_CTRL_FLAG_VOLATILE);
}

void V4L2Controls::V4L2Control::reset() {
    int ret;

    MV_HAL_LOG_TRACE() << "Resetting control" << name();
    switch (query_.type) {
        case V4L2_CTRL_TYPE_INTEGER:
            // default value should not change during runtime, no need to refresh the control.
            ret = set_int(query_.default_value);
            if (ret) {
                MV_HAL_LOG_ERROR() << "Failed to reset" << name() << "to default value" << query_.default_value;
            }
            break;

        case V4L2_CTRL_TYPE_BUTTON:
            break;

        default:
            struct v4l2_ext_controls ctrls = {};
            // get default value
            ctrls.which = V4L2_CTRL_WHICH_DEF_VAL;
            ctrls.count = 1;
            ctrls.controls = &ctrl;

            if (ioctl(fd_, VIDIOC_G_EXT_CTRLS, &ctrls)) {
                MV_HAL_LOG_ERROR() << "reset: error getting ext_ctrl" << query_.name << std::strerror(errno);
                return;
            }

            // apply as current
            ctrls.which = V4L2_CTRL_ID2WHICH(query_.id);
            if (ioctl(fd_, VIDIOC_S_EXT_CTRLS, &ctrls)) {
                MV_HAL_LOG_ERROR() << "reset: error setting ext_ctrl" << query_.name << std::strerror(errno);
                return;
            }

            break;
    }
}

V4L2Controls::V4L2Controls(int fd) : fd_(fd) {
    memset(&query_, 0, sizeof(query_));
    enumerate();
}

void V4L2Controls::enumerate() {
    V4L2Controls::V4L2Control ctrl(fd_);

    memset(&query_, 0, sizeof(query_));
    while ((get_next_control(ctrl) == 0)) {
        MV_HAL_LOG_TRACE() << "Control" << query_.name << "found";
        control_cache_.emplace(query_.name, ctrl);
    }

    // notify if not controls have been found:
    if (control_cache_.empty()) {
        MV_HAL_LOG_WARNING() << "No controls found";
    }
}

V4L2Controls::V4L2Control &V4L2Controls::get(const std::string &name) {
    int ret = 0;
    V4L2Controls::V4L2Control &control = control_cache_.at(name);
    ret = control.refresh();
    if (ret) {
        auto strerr = std::strerror(errno);
        throw std::runtime_error("Failed to refresh control " + name + " (" + strerr + "), ret = " + std::to_string(ret));
    }

    return control;
}

bool V4L2Controls::has(const std::string &name) const {
    for (const auto &p : control_cache_) {
        if (p.first.find(name) == 0) {
            return true;
        }
    }

    return false;
}

void V4L2Controls::foreach(std::function<int(V4L2Controls::V4L2Control &)> f) {
    for (auto &[name, ctrl] : control_cache_) {
        f(ctrl);
    }
}

int V4L2Controls::query() {
    struct v4l2_queryctrl qc;
    int rc;

    query_.id |= V4L2_CTRL_FLAG_NEXT_COMPOUND | V4L2_CTRL_FLAG_NEXT_CTRL;

    rc = ioctl(fd_, VIDIOC_QUERY_EXT_CTRL, &query_);
    if (rc != ENOTTY) {
        return rc;
    }

    qc.id = query_.id;
    rc = ioctl(fd_, VIDIOC_QUERYCTRL, &qc);
    if (rc == 0) {
        query_.type = qc.type;
        memcpy(query_.name, qc.name, sizeof(query_.name));
        query_.minimum = qc.minimum;
        if (qc.type == V4L2_CTRL_TYPE_BITMASK) {
            query_.maximum = static_cast<__u32>(qc.maximum);
            query_.default_value = static_cast<__u32>(qc.default_value);
        } else {
            query_.maximum = qc.maximum;
            query_.default_value = qc.default_value;
        }
        query_.step = qc.step;
        query_.flags = qc.flags;
        query_.elems = 1;
        query_.nr_of_dims = 0;
        memset(query_.dims, 0, sizeof(query_.dims));
        switch (query_.type) {
            case V4L2_CTRL_TYPE_INTEGER64:
                query_.elem_size = sizeof(__s64);
                break;
            case V4L2_CTRL_TYPE_STRING:
                query_.elem_size = qc.maximum + 1;
                query_.flags |= V4L2_CTRL_FLAG_HAS_PAYLOAD;
                break;
            default:
                query_.elem_size = sizeof(__s32);
                break;
        }
        memset(query_.reserved, 0, sizeof(query_.reserved));
    }

    query_.id = qc.id;
    return rc;
}

int V4L2Controls::get_next_control(V4L2Controls::V4L2Control &ctrl) {
    struct v4l2_ext_controls ctrls;
    int ret;

    memset(&ctrls, 0, sizeof(ctrls));
    memset(&ctrl.ctrl, 0, sizeof(ctrl.ctrl));

    do {
        ret = query();
    } while (ret == 0 && (query_.flags & V4L2_CTRL_FLAG_DISABLED || query_.type == V4L2_CTRL_TYPE_CTRL_CLASS));

    if (ret)
        return ret;

    // TODO: Check if needed HERE or BEFORE loop ??
    // save the query used to get this control
    memcpy(&ctrl.query_, &query_, sizeof(query_));

    ctrl.ctrl.id = query_.id;
    ctrls.which = V4L2_CTRL_ID2WHICH(query_.id);
    ctrls.count = 1;
    ctrls.controls = &ctrl.ctrl;

    if (query_.type == V4L2_CTRL_TYPE_INTEGER64 ||
        query_.type == V4L2_CTRL_TYPE_INTEGER ||
        query_.type == V4L2_CTRL_TYPE_STRING ||
        query_.nr_of_dims ||
        query_.type >= V4L2_CTRL_COMPOUND_TYPES ||
        (V4L2_CTRL_ID2WHICH(query_.id) != V4L2_CTRL_CLASS_USER &&
        query_.id < V4L2_CID_PRIVATE_BASE)) {

        if (query_.flags & V4L2_CTRL_FLAG_HAS_PAYLOAD) {
            ctrl.ctrl.size = query_.elems * query_.elem_size;
            ctrl.ctrl.ptr = calloc(1, ctrl.ctrl.size);
        } else {
            ctrl.ctrl.ptr = NULL;
        }

        if (ioctl(fd_, VIDIOC_G_EXT_CTRLS, &ctrls)) {
            MV_HAL_LOG_ERROR() << "error getting ext_ctrl" << ctrl.query_.name << std::strerror(errno);
            return -1;
        }

    } else {
        struct v4l2_control basic_ctrl;
        basic_ctrl.id = query_.id;
        if (query_.type != V4L2_CTRL_TYPE_BUTTON) {
            if (ioctl(fd_, VIDIOC_G_CTRL, &basic_ctrl)) {
                MV_HAL_LOG_ERROR() << "error getting ctrl" << query_.name << std::strerror(errno);
                return -1;
            }

            ctrl.ctrl.value = basic_ctrl.value;
            ctrl.ctrl.ptr = &ctrl.ctrl.value;
            ctrl.ctrl.size = sizeof(ctrl.ctrl.value);
        }
    }

    return ret;
}
} // namespace Metavision
