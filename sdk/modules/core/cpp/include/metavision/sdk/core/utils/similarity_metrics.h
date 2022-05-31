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

#ifndef METAVISION_SDK_ML_SIMILARITY_METRICS_H
#define METAVISION_SDK_ML_SIMILARITY_METRICS_H

#include <algorithm>
#include <vector>
#include <assert.h>

namespace Metavision {
namespace Utils {

/// @brief Computes intersection area of two boxes
/// @param box1 Description of a 2D box
/// @param box2 Description of a 2D box
/// @return proportion of intersection area
template<typename T1, typename T2>
inline float intersection(const T1 &box1, const T2 &box2) {
    decltype(box1.x) zero(0);
    float w_intersect = std::max(zero, std::min(box1.x + box1.w, box2.x + box2.w) - std::max(box1.x, box2.x));
    float h_intersect = std::max(zero, std::min(box1.y + box1.h, box2.y + box2.h) - std::max(box1.y, box2.y));
    return w_intersect * h_intersect;
}

/// @brief Computes the ratio between intersection area and union area
/// @param box1 Description of a 2D box
/// @param box2 Description of a 2D box
/// @return Proportion of intersection area
template<typename T1, typename T2>
inline float intersection_over_union(const T1 &box1, const T2 &box2) {
    float intersection_area = intersection(box1, box2);
    float union_area        = box1.w * box1.h + box2.w * box2.h - intersection_area;
    return union_area > 0 ? intersection_area / union_area : 0;
}

/// @brief Helper function to compute similarity, taking into account the class.
/// @param box1 Description of a 2D box
/// @param box2 Description of a 2D box
/// @return Proportion of intersection area if class ID are the same, 0 otherwise
template<typename T1, typename T2>
inline float compute_similarity_iou_using_classid(const T1 &box1, const T2 &box2) {
    if (box1.class_id == box2.class_id) {
        return Utils::intersection_over_union(box1, box2);
    }

    return 0.f;
}

/// Helper function to compute similarity, taking into account the class and a similarity matrix
/// @param box1 Description of a 2D box
/// @param box2 Description of a 2D box
/// @param similarity_matrix (nb_object_classes + 1) x (nb_object_classes + 1) matrix of similarity weights.
/// @param nb_object_classes Number of valid object classes. class_id 0 is 'background' class and is not counted as a
/// valid object class
/// @return Weighted proportion of intersection area (weights depends on class id)
template<typename T1, typename T2>
inline float compute_similarity_iou_using_classid_and_similarity_matrix(const T1 &box1, const T2 &box2,
                                                                        const std::vector<float> &similarity_matrix,
                                                                        unsigned int nb_object_classes) {
    const auto class_1 = box1.class_id;
    const auto class_2 = box2.class_id;
    assert(class_1 > 0);
    assert(class_1 <= nb_object_classes);
    assert(class_2 > 0);
    assert(class_2 <= nb_object_classes);
    assert(similarity_matrix.size() == (nb_object_classes + 1) * (nb_object_classes + 1));

    float similarity = Utils::intersection_over_union(box1, box2);
    similarity *= similarity_matrix[class_1 * (nb_object_classes + 1) + class_2];
    return similarity;
}

} // namespace Utils
} // namespace Metavision

#endif // METAVISION_SDK_ML_SIMILARITY_METRICS_H
