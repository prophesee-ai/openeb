#ifndef V4L2_CONTROLS_H
#define V4L2_CONTROLS_H

#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <cstring>
#include <cstdint>
#include <optional>
#include <map>
#include <string>
#include <functional>
#include <cstdint>
#include <linux/videodev2.h>
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/hal_log.h"


namespace Metavision {

class V4L2Controls {
    public:
    class V4L2Control {
    public:
        V4L2Control(int fd);
        std::string name() const { return query_.name; }
        std::string strtype() const;
        int type() const { return query_.type; }

        std::optional<int> get_int(void);
        std::optional<int64_t> get_int64(void);
        std::optional<bool> get_bool(void);
        std::optional<std::string> get_str(void);

        [[nodiscard]] int set_int(int value);
        [[nodiscard]] int set_int64(std::int64_t value);
        [[nodiscard]] int set_bool(bool value);
        [[nodiscard]] int push();

        void reset();
        bool is_volatile();

        template <typename T>
        std::optional<T*> get_compound() {
            // guard
            if (query_.type < V4L2_CTRL_COMPOUND_TYPES) {
                MV_HAL_LOG_ERROR() << "Control is not of compound type";
                return {};
            }

            if (sizeof(T) != ctrl.size) {
                MV_HAL_LOG_ERROR() << "Control payload size does not match structure";
                MV_HAL_LOG_ERROR() << "Control payload size: " << ctrl.size << " structure size: " << sizeof(T);

                return {};
            }

            refresh();

            // TODO: refresh
            return {reinterpret_cast<T*>(ctrl.ptr)};
        }

        template <typename T>
        [[nodiscard]] int set_compound(T *value) {
            // guard
            if (query_.type < V4L2_CTRL_COMPOUND_TYPES) {
                return -EINVAL;
            }

            if (sizeof(T) != ctrl.size) {
                return -EINVAL;
            }

            memcpy(ctrl.ptr, value, sizeof(T));
            apply();

            return 0;
        }

        int apply() {
            int ret = 0;
            struct v4l2_ext_controls ctrls;
            ctrls.which = V4L2_CTRL_ID2WHICH(query_.id);
            ctrls.count = 1;
            ctrls.controls = &ctrl;
            if ((ret = ioctl(fd_, VIDIOC_S_EXT_CTRLS, &ctrls)) < 0) {
                MV_HAL_LOG_ERROR() << "Failed to update v4l2 control" << query_.name << std::strerror(errno);
                return -1;
            }

            return 0;
        }

        int refresh() {
            if (query_.type == V4L2_CTRL_TYPE_BUTTON) {
                return 0;
            }

            struct v4l2_ext_controls ctrls = {0};
            ctrls.which = V4L2_CTRL_ID2WHICH(query_.id);
            ctrls.count = 1;
            ctrls.controls = &ctrl;
            return ioctl(fd_, VIDIOC_G_EXT_CTRLS, &ctrls);
        }

        template <typename T>
        std::optional<T> get_min() {
            // only applicable to integer types
            if (query_.type != V4L2_CTRL_TYPE_INTEGER || query_.type != V4L2_CTRL_TYPE_INTEGER64) {
                MV_HAL_LOG_ERROR() << "Control is not of integer type";
                return {};
            }

            if (sizeof(T) != ctrl.size) {
                MV_HAL_LOG_ERROR() << "Control payload size does not match structure";
                return {};
            }

            return *reinterpret_cast<T*>(&query_.minimum);
        }

        template <typename T>
        std::optional<T> get_max() {
            if (query_.type != V4L2_CTRL_TYPE_INTEGER || query_.type != V4L2_CTRL_TYPE_INTEGER64) {
                MV_HAL_LOG_ERROR() << "Control is not of integer type";
                return {};
            }

            if (sizeof(T) != ctrl.size) {
                MV_HAL_LOG_ERROR() << "Control payload size does not match structure";
                return {};
            }

            return *reinterpret_cast<T*>(&query_.maximum);
        }

        v4l2_ext_control ctrl;
        v4l2_query_ext_ctrl query_;
            
    private:
        int fd_;
    };

    V4L2Controls(int fd);

    void enumerate();

    V4L2Control& get(const std::string &name);
    bool has(const std::string &name) const;

    void foreach(std::function<int(V4L2Control &)> f);
private:
    // controls cache (map)
    std::map<std::string, V4L2Control> control_cache_;
    std::map<std::string, bool> dirt_map_;
    struct v4l2_query_ext_ctrl query_;

    int query();

    int get_next_control(V4L2Control &ctrl);

    int fd_;
};
} // namespace Metavision

#endif // V4L2_CONTROLS_H
