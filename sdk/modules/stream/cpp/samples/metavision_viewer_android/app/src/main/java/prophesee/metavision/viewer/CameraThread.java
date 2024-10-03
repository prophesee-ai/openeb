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

package prophesee.metavision.viewer;

public class CameraThread extends Thread {
    private boolean stop_ = false;

    public CameraThread(String path) {
        stop_ = false;

        NativeHelper.createCameraFromRaw(path);
    }

    public void run() {
        stop_ = false;

        NativeHelper.startCamera();
        while (true) {
            if (stop_) {
                NativeHelper.stopCamera();
                break;
            }
        }
    }

    public void shutdown() {
        stop_ = true;
    }
}

