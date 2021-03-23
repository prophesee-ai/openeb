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

#ifndef METAVISION_UTILS_GTEST_WITH_TMP_DIR_H
#define METAVISION_UTILS_GTEST_WITH_TMP_DIR_H

#include <memory>
#include <gtest/gtest.h>

#include "metavision/utils/gtest/temporary_directory_handler.h"

namespace Metavision {

class GTestWithTmpDir : virtual public ::testing::Test {
public:
    GTestWithTmpDir() : tmpdir_handler_(new TemporaryDirectoryHandler("temporary_directory_for_gtests")) {}

    virtual ~GTestWithTmpDir() {}

protected:
    // The temporary directory handler is held through a unique pointer so that it can be reset
    // to a different path at any point during the lifetime of the gtest.
    // Note that it can't be done if the handler is stored as a value, since the temporary
    // directly handler cannot (and should not) be copied.
    std::unique_ptr<TemporaryDirectoryHandler> tmpdir_handler_;
};

} // namespace Metavision

#endif // METAVISION_UTILS_GTEST_WITH_TMP_DIR_H
