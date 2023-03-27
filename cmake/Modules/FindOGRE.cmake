# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

find_package(OGRE QUIET COMPONENTS Bites Overlay CONFIG)

if(NOT OGRE_FOUND) 
    return()
endif()

set(OGRE_FOUND ${OGRE_FOUND} CACHE INTERNAL "Ogre lib found on your system")
set(OGRE_LINK_LIBS OgreBites)

# When complied from source, Ogre3d comes with Imgui. But when installed from packages, 
# Imgui might not be set as dependencies. Because we rely imgui, let's make sure we find it
# on the system
find_package(imgui QUIET)

if(imgui_FOUND OR TARGET imgui::imgui) 
    list(APPEND OGRE_LINK_LIBS imgui::imgui)
else()
    get_target_property(ogre_overlay_include_dir OgreOverlay INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT EXISTS ${ogre_overlay_include_dir}/imgui.h)
        find_library(imgui_lib imgui)
        find_file(imgui_header imgui.h PATH_SUFFIXES imgui)
        get_filename_component(imgui_header_dir ${imgui_header} DIRECTORY)

        if(imgui_lib MATCHES "NOTFOUND")
            message(NOTICE "ImGui library not found, or Ogr3d not compiled with Overlay Imgui")
            return()
        endif() 
        if(imgui_header MATCHES "NOTFOUND") 
            message(NOTICE "ImGui headers not found, or Ogr3d not compiled with Overlay Imgui")
            return()
        endif() 

        add_library(imgui UNKNOWN IMPORTED GLOBAL) 
        set_target_properties(imgui PROPERTIES 
            IMPORTED_LOCATION ${imgui_lib}
            IMPORTED_LINK_INTERFACE_LIBRARIES "freetype;stb"
            INTERFACE_INCLUDE_DIRECTORIES ${imgui_header_dir}
        )
        add_library(imgui::imgui ALIAS imgui)
        list(APPEND OGRE_LINK_LIBS imgui::imgui)
    endif()
    set(imgui_FOUND TRUE) 
endif()
