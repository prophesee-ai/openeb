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

#ifndef METAVISION_SDK_CORE_FRAME_COMPOSER_H
#define METAVISION_SDK_CORE_FRAME_COMPOSER_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "metavision/sdk/base/utils/timestamp.h"
#include "metavision/sdk/base/utils/sdk_log.h"

namespace Metavision {

/// @brief This class implements an object capable of displaying a big image containing many images in an openGL
/// window.
///
/// You need to first call method @ref FrameComposer::add_new_subimage_parameters for each small image that you want to
/// display in the big one. It will create an ImageParams object that will store the reference to the small image but
/// also its position and other fields. Every time you call @ref FrameComposer::update_subimage, FrameComposer will
/// gather all ImageParams, put image at the right place in the big one and copy that in the output image.
///
/// Ideally the class takes integer pixels as inputs (CV_8UC1 or CV_8UC3). In case of floating point values, the cv::Mat
/// is converted to an int type, which might slightly slow down the process.
class FrameComposer {
public:
    enum InterpolationType { Nearest = cv::INTER_NEAREST, Linear = cv::INTER_LINEAR, Area = cv::INTER_AREA };

    /// @brief Settings used to resize or crop an image
    struct ResizingOptions {
        ResizingOptions(unsigned int width, unsigned int height, bool enable_crop = false,
                        InterpolationType interp_type = Nearest) :
            width(width), height(height), enable_crop(enable_crop), interp_type(interp_type) {}

        const unsigned int width, height;    ///< New size of the image in the final composition
        const bool enable_crop;              ///< Whether to enable cropping the image to the specified width and height
                                             /// (maintains the center)
        const InterpolationType interp_type; ///< Type of interpolation used to resize images
    };

    /// @brief Settings used to rescale and/or apply a colormap on a grayscale image
    struct GrayToColorOptions {
        GrayToColorOptions() = default;

        unsigned char min_rescaling_value = 0;   ///< Min grayscale value for the rescaling
        unsigned char max_rescaling_value = 255; ///< Max grayscale value for the rescaling
        // (Keep original values if min == max)

        int color_map_id = -1; ///< Colormap to apply:
                               /// - None             = -1
                               /// - COLORMAP_AUTUMN  = 0
                               /// - COLORMAP_BONE    = 1
                               /// - COLORMAP_JET     = 2
                               /// - COLORMAP_WINTER  = 3
                               /// - COLORMAP_RAINBOW = 4
                               /// - COLORMAP_OCEAN   = 5
                               /// - COLORMAP_SUMMER  = 6
                               /// - COLORMAP_SPRING  = 7
                               /// - COLORMAP_COOL    = 8
                               /// - COLORMAP_HSV     = 9
                               /// - COLORMAP_PINK    = 10
                               /// - COLORMAP_HOT     = 11
    };

    /// @brief Constructor
    /// @param background_color Default color of the composition background (CV_8UC3)
    FrameComposer(const cv::Vec3b &background_color = cv::Vec3b(0, 0, 0));

    /// @brief Destructor
    ~FrameComposer();

    /// @brief Selects a sub-part of the final composed image and defines its preprocessing options
    /// @warning It's possible to define sub-images with overlapping regions of interests. Each update of a
    /// sub-image overwrites the corresponding subpart of the image, and thus also overwrites the overlapping areas.
    /// Thus, if a small sub-image is located on top of a larger sub-image, it will only be visible if its resfresh rate
    /// is larger than or equal to the one of the larger sub-image, and ist update with @ref update_subimage is called
    /// at the end
    /// @param top_left_x X-position of the top-left corner of the image in the composition
    /// @param top_left_y Y-position of the top-left corner of the image in the composition
    /// @param resize_options Options used to resize the image
    /// @param gray_to_color_options Options used to rescale and/or apply a colormap on the grayscale image
    /// @return Reference (integer) of the sub-image to update later with @ref update_subimage
    unsigned int add_new_subimage_parameters(int top_left_x, int top_left_y, const ResizingOptions &resize_options,
                                             const GrayToColorOptions &gray_to_color_options);

    /// @brief Update the sub-part of the final image corresponding to the reference @p img_ref
    /// @param img_ref Reference ID of the sub-image, defined in the function @ref add_new_subimage_parameters
    /// @param image Image to display at the corresponding location in the full composed image
    /// If the type is different than CV_8UC3 or CV_8UC1, then a conversion is performed
    bool update_subimage(unsigned int img_ref, const cv::Mat &image);

    /// @brief Gets the full image
    /// @return The composed image
    const cv::Mat &get_full_image() const;

    /// @brief Gets width of the output image
    /// @return The width of the output image
    const int get_total_width() const;

    /// @brief Gets height of the output image
    /// @return The height of the output image
    const int get_total_height() const;

private:
    /// @brief Struct containing an image alongside with its preprocessing options and relative position inside the
    /// final composed image
    struct ImageParams {
        cv::Mat image, image_grey_tmp, image_copy_tmp;
        cv::Point position;
        cv::Size size;
        cv::Rect roi;
        int cmap;
        InterpolationType interp_type;
        bool enable_crop;
        cv::Mat rescaling_lut;
    };

    /// @brief Fits the size of the FrameComposer canvas to the minimum englobing size, after adding a new subimage
    /// @note It will automatically be called after adding a new subimage slot
    /// @param new_subimage_id ID of the subimage that has just been added
    void fit_size(const unsigned int new_subimage_id);

    /// @brief Processes the source image @p src according to the setting from @p params and saves it inside @p params
    ///
    /// If needed, the grey image is rescaled between the two extreme intensity values defined in @p params
    ///
    /// @param src CV_8U or CV_8UC3 original input image
    /// @param params Struct containing the subimage alongside with its preprocessing options and relative position
    /// inside the final composed image
    void img_convert(const cv::Mat &src, ImageParams &params);

    /// @brief Applies a colormap on the grey image @p src and saves it inside @p params
    /// @param src CV_8U original input image
    /// @param params Struct containing the subimage alongside with its preprocessing options and relative position
    /// inside the final composed image
    void grey_to_bgr(const cv::Mat &src, ImageParams &params);

    unsigned int width_, height_;
    cv::Vec3b back_color_;
    std::vector<ImageParams> srcs_;
    cv::Mat full_img_;
};

inline FrameComposer::FrameComposer(const cv::Vec3b &background_color) :
    width_(0), height_(0), back_color_(background_color) {}

inline FrameComposer::~FrameComposer() {}

inline unsigned int FrameComposer::add_new_subimage_parameters(int top_left_x, int top_left_y,
                                                               const ResizingOptions &resize_options,
                                                               const GrayToColorOptions &gray_to_color_options) {
    ImageParams params;

    // Resizing
    params.position    = cv::Point(top_left_x, top_left_y);
    params.roi         = cv::Rect(top_left_x, top_left_y, resize_options.width, resize_options.height);
    params.size        = cv::Size(resize_options.width, resize_options.height);
    params.enable_crop = resize_options.enable_crop;
    params.interp_type = resize_options.interp_type;

    // Rescaling and Colormap
    params.cmap                  = gray_to_color_options.color_map_id; // Negative means no colormap
    const unsigned char &min_val = gray_to_color_options.min_rescaling_value;
    const unsigned char &max_val = gray_to_color_options.max_rescaling_value;

    if (min_val < max_val && (min_val != 0 || max_val != 255)) {
        const double scale  = 255. / (max_val - min_val);
        const double offset = -min_val * scale;

        params.rescaling_lut = cv::Mat(1, 256, CV_8UC1);
        uchar *p             = params.rescaling_lut.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = std::min(255, std::max(0, int(offset + scale * i))); // clamp
        }
    }
    srcs_.push_back(params);
    const unsigned int id = static_cast<unsigned int>(srcs_.size() - 1);
    fit_size(id); // Update the size of the composed image if needed
    return id;
}

inline bool FrameComposer::update_subimage(unsigned int img_ref, const cv::Mat &image) {
    if (img_ref >= srcs_.size() || image.empty())
        return false;

    ImageParams &params = srcs_[img_ref];

    // Ideally the class takes integer pixels as inputs
    // In case of floating point values, they're converted to integers
    if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
        switch (image.channels()) {
        case 1:
            image.convertTo(params.image_copy_tmp, CV_8UC1, 255.);
            break;
        case 3:
            image.convertTo(params.image_copy_tmp, CV_8UC3, 255.);
            break;
        default:
            MV_SDK_LOG_ERROR() << "Frames must have either 1 or 3 channels, not " << image.channels();
            return false;
        }
        return update_subimage(img_ref, params.image_copy_tmp);
    }

    // Resize if needed
    if (image.size() == params.size)
        img_convert(image, params);
    else if (params.enable_crop && image.cols >= params.size.width && image.rows >= params.size.height) {
        const cv::Rect crop_rect((image.cols - params.size.width) / 2, (image.rows - params.size.height) / 2,
                                 params.size.width, params.size.height);
        img_convert(image(crop_rect), params);
    } else {
        cv::resize(image, params.image, params.size, 0., 0., params.interp_type);
        if (image.channels() != 3) // Otherwise the color image is already in params.image
            img_convert(params.image, params);
    }
    assert(!params.image.empty());
    return true;
}

inline void FrameComposer::img_convert(const cv::Mat &src, ImageParams &params) {
    if (src.channels() == 3) {
        src.copyTo(params.image);
        return;
    }

    assert(src.channels() == 1);
    // Apply Rescaling if needed
    if (params.rescaling_lut.empty())
        grey_to_bgr(src, params);
    else {
        cv::LUT(src, params.rescaling_lut, params.image_grey_tmp);
        grey_to_bgr(params.image_grey_tmp, params);
    }
}

inline void FrameComposer::grey_to_bgr(const cv::Mat &src, ImageParams &params) {
    // Apply colormap if needed
    if (params.cmap >= 0)
        cv::applyColorMap(src, params.image, params.cmap);
    else
        cv::cvtColor(src, params.image, cv::COLOR_GRAY2BGR, 3);
}

inline void FrameComposer::fit_size(const unsigned int new_subimage_id) {
    const ImageParams &params = srcs_[new_subimage_id];
    unsigned int lw           = params.position.x + params.size.width;
    unsigned int lh           = params.position.y + params.size.height;

    // Reallocate memory if needed
    if (lw > width_ || lh > height_) {
        width_  = std::max(width_, lw);
        height_ = std::max(height_, lh);
        if (full_img_.empty()) {
            full_img_.create(height_, width_, CV_8UC3);
            full_img_.setTo(back_color_);
        } else {
            cv::Mat tmp = full_img_.clone();
            const cv::Rect original_roi(0, 0, full_img_.cols, full_img_.rows);

            full_img_.create(height_, width_, CV_8UC3);
            full_img_.setTo(back_color_);
            tmp.copyTo(full_img_(original_roi));
        }
    }

    // Use ROIs as references to the full composed image
    for (auto &params : srcs_)
        params.image = full_img_(params.roi);
}

inline const cv::Mat &FrameComposer::get_full_image() const {
    return full_img_;
}

inline const int FrameComposer::get_total_width() const {
    return full_img_.cols;
}

inline const int FrameComposer::get_total_height() const {
    return full_img_.rows;
}

} // namespace Metavision

#endif // METAVISION_SDK_CORE_FRAME_COMPOSER_H
