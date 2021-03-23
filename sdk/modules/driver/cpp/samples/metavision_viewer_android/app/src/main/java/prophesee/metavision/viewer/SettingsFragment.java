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

import android.os.Bundle;
import android.preference.PreferenceFragment;
import android.preference.ListPreference;
import android.preference.Preference;
import android.widget.ListView;
import android.view.ViewGroup.OnHierarchyChangeListener;
import android.view.View;
import android.app.AlertDialog;

public class SettingsFragment extends PreferenceFragment {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Load the preferences from an XML resource
        addPreferencesFromResource(R.xml.preferences);

        getPreferenceScreen().getSharedPreferences()
                .registerOnSharedPreferenceChangeListener(MainActivity.getPreferenceListener());


        /*
        ListPreference frameTypePreference = (ListPreference) findPreference("frame_type_list_preference");

        // hack to disable selection of EMs and CDs + EMs for the moment
        frameTypePreference.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener() {

            public boolean onPreferenceClick(Preference preference) {
                final ListPreference listPref = (ListPreference) preference;
                ListView listView = ((AlertDialog)listPref.getDialog()).getListView();
                listView.setOnHierarchyChangeListener(new OnHierarchyChangeListener() {

                    // assuming list entries are created in the order of the entry values
                    int counter = 0;

                    public void onChildViewRemoved(View parent, View child) {}

                    public void onChildViewAdded(View parent, View child) {
                        String key = listPref.getEntryValues()[counter].toString();
                        if (key.equals("2") || key.equals("1")) {
                            child.setEnabled(false);
                        }
                        counter++;
                    }
                });
                return false;
            }
        });
        */
    }
}
