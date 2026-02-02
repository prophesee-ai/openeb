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

import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.preference.PreferenceManager;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

//import com.google.android.gms.common.api.GoogleApiClient;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {
    static TextView textview_;
    static MainActivity mainactivity_;
    static Boolean preference_initialized_ = false;
    static String raw_path_, hal_plugin_path_;

    static SharedPreferences.OnSharedPreferenceChangeListener preference_listener_ = new SharedPreferences.OnSharedPreferenceChangeListener() {
        public void onSharedPreferenceChanged(SharedPreferences prefs, String key) {
            // listener implementation
            if (key.equals("text_message")) {
                if (prefs.getBoolean(key, true)) {
                    if (textview_ != null) {
                        textview_.setVisibility(View.VISIBLE);
                    }
                } else {
                    if (textview_ != null) {
                        textview_.setVisibility(View.GONE);
                    }
                }
            }
        }

    };
    private static boolean forceClaim = true;
    static private CameraThread camera_thread_;

    Timer timer_;
    long time_ = 0;
    long last_cd_counter_ = 0, last_em_counter_ = 0;

    private CameraFrameView tdview_;

    /**
     * ATTENTION: This was auto-generated to implement the App Indexing API. See
     * https://g.co/AppIndexing/AndroidStudio for more information.
     */
    // private GoogleApiClient client;
    static SharedPreferences.OnSharedPreferenceChangeListener getPreferenceListener() {
        return preference_listener_;
    }

    private static boolean copyAssetFolder(AssetManager assetManager, String fromAssetPath, String toPath) {
        try {
            String[] files = assetManager.list(fromAssetPath);
            new File(toPath).mkdirs();
            boolean res = true;
            for (String file : files)
                if (file.contains("."))
                    res &= copyAsset(assetManager, fromAssetPath + "/" + file, toPath + "/" + file);
                else
                    res &= copyAssetFolder(assetManager, fromAssetPath + "/" + file, toPath + "/" + file);
            return res;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static boolean copyAsset(AssetManager assetManager, String fromAssetPath, String toPath) {
        InputStream in = null;
        OutputStream out = null;
        try {
            in = assetManager.open(fromAssetPath);
            new File(toPath).createNewFile();
            out = new FileOutputStream(toPath);
            copyFile(in, out);
            in.close();
            in = null;
            out.flush();
            out.close();
            out = null;
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.preferences: {
                Intent intent = new Intent();
                intent.setClassName(this, "prophesee.metavision.viewer.PreferenceActivity");
                startActivity(intent);
                return true;
            }
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        try {
            requestWindowFeature(Window.FEATURE_NO_TITLE);
        } catch (Exception e) {

        }
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_main);
        init();
        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        // client = new GoogleApiClient.Builder(this).addApi(AppIndex.API).build();

        timer_ = new Timer();
        timer_.schedule(new TimerTask() {
            @Override
            public void run() {
                try {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            updateEventAtis();
                        }
                    });

                } catch (Exception e) {

                }
            }
        }, 0, 1000);// Update text ev
    }

    private String getInstallPath() {
        String toPath = "/data/data/" + getPackageName();
        return toPath;
    }

    void init() {
        mainactivity_ = this;

        AssetManager assetManager = getAssets();
        String toPath = getInstallPath();
        raw_path_ = toPath + "/gen4_evt3_hand.raw";
        hal_plugin_path_ = toPath + "/hal";
        try {
            copyAsset(assetManager, "gen4_evt3_hand.raw", raw_path_);
            copyAssetFolder(assetManager, "hal", hal_plugin_path_);
        } catch (Exception e) {
            Log.e("ASSETCOPY", "Except " + e.toString() + " " + e.getMessage());
        }
        NativeHelper.setupEnvironment(hal_plugin_path_ + "/plugins");

        /*
        File directory = new File(getInstallPath());
        File[] files = directory.listFiles();
        */

        tdview_ = (CameraFrameView) findViewById(R.id.tdview);//
        Button btnRaw = (Button) findViewById(R.id.buttonRaw);
        final TextView t = (TextView) findViewById(R.id.textView);
        textview_ = t;
        textview_.setVisibility(View.GONE);
        final MainActivity mainactivity = this;

        SharedPreferences sharedPref = PreferenceManager.getDefaultSharedPreferences(this);
        if (preference_initialized_ == false) {
            if (sharedPref.contains("text_message")) {
                SharedPreferences.Editor editor = sharedPref.edit();
                editor.putBoolean("text_message", false);
                editor.apply();
            }
            preference_initialized_ = true;
        }
        if (sharedPref.contains("text_message")) {
            boolean b = sharedPref.getBoolean("text_message", true);
            if (textview_ != null) {
                if (b) {
                    textview_.setVisibility(View.VISIBLE);
                } else {
                    textview_.setVisibility(View.GONE);
                }
            }
        }

        if (camera_thread_ != null) {
            setTDView();
        }

        btnRaw.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (ContextCompat.checkSelfPermission(mainactivity_, android.Manifest.permission.READ_EXTERNAL_STORAGE) !=
                        PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(mainactivity_, new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
                }

                if (camera_thread_ != null && camera_thread_.isAlive()) {
                    Log.d("b1", "camera_thread_ alive!");
                    return;
                }
                if (camera_thread_ != null) {
                    camera_thread_.shutdown();
                    try {
                        camera_thread_.join();
                    } catch (Exception e) {
                    }
                    camera_thread_ = null;
                }
                camera_thread_ = new CameraThread(raw_path_);
                if (tdview_ != null) {
                    int[] geometry = NativeHelper.getCameraGeometry();
                    tdview_.init(geometry[0], geometry[1]);
                }
                camera_thread_.start();
                setTDView();
            }
        });
    }

    String humandReadableEventRate(long ev_rate) {
        if (ev_rate > 1000000) {
            return Double.toString(Math.round((ev_rate / 1000000.) * 100) / 100.) + " MEv/s";
        } else if (ev_rate > 1000) {
            return Double.toString(Math.round((ev_rate / 1000.) * 100) / 100.) + " kEv/s";
        }
        return Double.toString((ev_rate * 100) / 100.) + " Ev/s";
    }

    void updateEventAtis() {
        if (time_ == 0) {
            time_ = System.currentTimeMillis();
        }

        long tmp = System.currentTimeMillis();
        if (tmp - time_ > 1000) {
            if (textview_ != null) {
                long[] counters = NativeHelper.getNumberOfEventsSinceStart();
                textview_.setText("");
                textview_.append("#CD : " + Long.toString(counters[0]));
                // Only for CD + EM camera
                // textview_.append(" #EM : " + Long.toString(counters[1]));
                if (last_cd_counter_ > 0) {
                    long num_cd = counters[0] - last_cd_counter_;
                    textview_.append(" CD rate : " + humandReadableEventRate(num_cd));
                }
                last_cd_counter_ = counters[0];
            }
            time_ = tmp;
        }
    }

    void setTDView() {
        tdview_.setBitmapUpdater(new CameraFrameView.BitmapUpdater() {
            @Override
            public void update(Bitmap bmp) {
                NativeHelper.updateFrame(bmp);

            }
        });
    }
}
