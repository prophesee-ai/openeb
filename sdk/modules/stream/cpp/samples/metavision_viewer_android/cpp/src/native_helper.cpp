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

#include <ostream>
#include <streambuf>
#include <cstdlib>
#include <string.h>
#include <jni.h>
#include <dirent.h>
#include <dlfcn.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#if CV_MAJOR_VERSION >= 4
#include <opencv2/imgproc/types_c.h>
#endif
#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/utils/cd_frame_generator.h>

extern "C" {

namespace {
Metavision::Camera camera;
bool camera_initialized = false;
std::unique_ptr<Metavision::CDFrameGenerator> cd_frame_generator;
cv::Mat frame;
std::int64_t callback_id = -1;
int frame_type;
std::uint64_t cd_counter = 0, em_counter = 0;

class AndroidLogPrintStreamBuf : public std::streambuf {
    enum {
        BUFFER_SIZE = 255,
    };

public:
    AndroidLogPrintStreamBuf() {
        buffer_[BUFFER_SIZE] = '\0';
        setp(buffer_, buffer_ + BUFFER_SIZE - 1);
    }

    ~AndroidLogPrintStreamBuf() {
        sync();
    }

protected:
    virtual int_type overflow(int_type c) {
        if (c != EOF) {
            *pptr() = c;
            pbump(1);
        }
        flush_buffer();
        return c;
    }

    virtual int sync() {
        flush_buffer();
        return 0;
    }

private:
    int flush_buffer() {
        int len = int(pptr() - pbase());
        if (len <= 0)
            return 0;

        if (len <= BUFFER_SIZE)
            buffer_[len] = '\0';

        android_LogPriority t = ANDROID_LOG_VERBOSE;
        __android_log_write(ANDROID_LOG_VERBOSE, "", buffer_);

        pbump(-len);
        return len;
    }

private:
    char buffer_[BUFFER_SIZE + 1];
};

class AndroidLogPrintOutputStream : public std::ostream {
public:
    AndroidLogPrintOutputStream() : std::ostream(new AndroidLogPrintStreamBuf()){};

    virtual ~AndroidLogPrintOutputStream() {
        delete rdbuf();
    }
};
std::unique_ptr<AndroidLogPrintOutputStream> android_log_stream;
} // namespace

JNIEXPORT void JNICALL Java_prophesee_metavision_viewer_NativeHelper_setupEnvironment(JNIEnv *env, jobject thiz,
                                                                                      jstring HAL_plugin_path) {
    // setup HAL plugin environment variable
    const char *HAL_plugin_path_cstr = env->GetStringUTFChars(HAL_plugin_path, JNI_FALSE);
    std::string libpath              = std::string(HAL_plugin_path_cstr) + "/../../lib";
    setenv("MV_HAL_PLUGIN_PATH", HAL_plugin_path_cstr, 1);
    env->ReleaseStringUTFChars(HAL_plugin_path, HAL_plugin_path_cstr);

    // setup log level and stream so that metavision logs are redirected to android logcat
    setenv("MV_LOG_LEVEL", "TRACE", 1);
    android_log_stream.reset(new AndroidLogPrintOutputStream);
    Metavision::setLogStream(*android_log_stream);
}

JNIEXPORT jlongArray JNICALL Java_prophesee_metavision_viewer_NativeHelper_getNumberOfEventsSinceStart(JNIEnv *env,
                                                                                                       jobject thiz) {
    jlongArray newArray = env->NewLongArray(2);

    jlong *narr = env->GetLongArrayElements(newArray, NULL);
    if (camera_initialized) {
        narr[0] = cd_counter;
        narr[1] = em_counter;
    }
    env->ReleaseLongArrayElements(newArray, narr, 0);

    return newArray;
}

JNIEXPORT jintArray JNICALL Java_prophesee_metavision_viewer_NativeHelper_getCameraGeometry(JNIEnv *env, jobject thiz) {
    jintArray newArray = env->NewIntArray(2);

    jint *narr = env->GetIntArrayElements(newArray, NULL);
    if (camera_initialized) {
        auto &geometry = camera.geometry();
        narr[0]        = geometry.get_width();
        narr[1]        = geometry.get_height();
    }
    env->ReleaseIntArrayElements(newArray, narr, 0);

    return newArray;
}

JNIEXPORT jboolean JNICALL Java_prophesee_metavision_viewer_NativeHelper_createCameraFromLive(JNIEnv *env,
                                                                                              jobject thiz) {
    if (camera_initialized)
        return true;
    try {
        camera = Metavision::Camera::from_first_available();

        cd_frame_generator.reset(
            new Metavision::CDFrameGenerator(camera.geometry().get_width(), camera.geometry().get_height()));

        camera.cd().add_callback([](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
            cd_counter += std::distance(ev_begin, ev_end);
            cd_frame_generator->add_events(ev_begin, ev_end);
        });

        cd_frame_generator->start(30, [](const Metavision::timestamp &, const cv::Mat &f) { frame = f; });
    } catch (Metavision::CameraException &e) {
        std::cerr << e.what() << std::endl;
        return false;
    }

    camera_initialized = true;
    return true;
}

JNIEXPORT jboolean JNICALL Java_prophesee_metavision_viewer_NativeHelper_createCameraFromRaw(JNIEnv *env, jobject thiz,
                                                                                             jstring path) {
    if (camera_initialized)
        return true;
    try {
        const char *raw_path_cstr = env->GetStringUTFChars(path, JNI_FALSE);
        camera                    = Metavision::Camera::from_file(std::string(raw_path_cstr));

        cd_frame_generator.reset(
            new Metavision::CDFrameGenerator(camera.geometry().get_width(), camera.geometry().get_height()));

        camera.cd().add_callback([](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
            cd_counter += std::distance(ev_begin, ev_end);
            cd_frame_generator->add_events(ev_begin, ev_end);
        });

        cd_frame_generator->start(30, [](const Metavision::timestamp &t, const cv::Mat &f) { frame = f; });

        env->ReleaseStringUTFChars(path, raw_path_cstr);
    } catch (Metavision::CameraException &e) {
        std::cerr << e.what() << std::endl;
        return false;
    }

    camera_initialized = true;
    return true;
}

JNIEXPORT jboolean JNICALL Java_prophesee_metavision_viewer_NativeHelper_isCameraRunnning(JNIEnv *env, jobject thiz) {
    if (!camera_initialized)
        return false;
    try {
        return camera.is_running();
    } catch (Metavision::CameraException &e) { std::cerr << e.what() << std::endl; }
    return false;
}

JNIEXPORT jboolean JNICALL Java_prophesee_metavision_viewer_NativeHelper_startCamera(JNIEnv *env, jobject thiz) {
    if (!camera_initialized)
        return false;
    try {
        if (!camera.start())
            return false;
    } catch (Metavision::CameraException &e) { std::cerr << e.what() << std::endl; }
    return true;
}

JNIEXPORT jboolean JNICALL Java_prophesee_metavision_viewer_NativeHelper_stopCamera(JNIEnv *env, jobject thiz) {
    if (!camera_initialized)
        return false;
    try {
        if (!camera.stop())
            return false;
    } catch (Metavision::CameraException &e) { std::cerr << e.what() << std::endl; }
    return true;
}

namespace {

void mat_to_bitmap(JNIEnv *env, const cv::Mat &src, jobject &bitmap) {
    void *pixels = 0;
    AndroidBitmap_lockPixels(env, bitmap, &pixels);

    cv::Mat tmp(src.rows, src.cols, CV_8UC4, pixels);
    if (src.type() == CV_8UC1) {
        cvtColor(src, tmp, CV_GRAY2RGBA);
    } else if (src.type() == CV_8UC3) {
        cvtColor(src, tmp, CV_BGR2RGBA);
    } else if (src.type() == CV_8UC4) {
        cvtColor(src, tmp, CV_BGRA2RGBA);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}

} // namespace

JNIEXPORT jlong JNICALL Java_prophesee_metavision_viewer_NativeHelper_updateFrame(JNIEnv *env, jobject thiz,
                                                                                  jobject bitmap) {
    if (!camera_initialized) {
        return 0;
    }
    if (!frame.empty()) {
        mat_to_bitmap(env, frame, bitmap);
    }
    return 0;
}
}
