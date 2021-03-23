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

import android.graphics.Bitmap;

public class NativeHelper {
    static {
        System.loadLibrary("metavision_viewer");
    }

    /**
     * Set the necessary environment to call native JNI code from Android.
     * This must be called once before the Camera API can be used.
     *
     * @param HAL_plugin_path path where all the HAL plugin path are installed
     */
    public static native void setupEnvironment(String HAL_plugin_path);

    /**
     * Creates and initializes a native Metavision::Camera from a live camera
     */
    public static native boolean createCameraFromLive();

    /**
     * Creates and initializes a native Metavision::Camera from a RAW file
     */
    public static native boolean createCameraFromRaw(String path);

    /**
     * Returns the number of CD and EM events received since the camera was started.
     */
    public static native long[] getNumberOfEventsSinceStart();

    /**
     * Returns the auto-detected geometry of the Metavision::Camera created by @ref createCamera
     * @return an array containing the width and height
     */
    public static native int[] getCameraGeometry();

    /**
     * Get the camera status
     */
    public static native boolean isCameraRunnning();

    /**
     * Starts the camera thread
     */
    public static native boolean startCamera();

    /**
     * Stops the camera thread
     */
    public static native boolean stopCamera();

    /**
     * Updates an Android bitmap from native C++ code created from received camera events
     * This must be called regularly to update the frame displayed in the app.
     */
    public static native void updateFrame(Bitmap bitmap);
}
