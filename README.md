# OpenEB

OpenEB is the open source project associated with [Metavision SDK](https://docs.prophesee.ai/stable/index.html)

It enables anyone to get a better understanding of event-based vision, directly interact with events and build
their own applications or camera plugins. As a camera manufacturer, ensure your customers benefit from the most advanced
event-based software suite available by building your own plugin. As a creator, scientist, academic, join and contribute
to the fast-growing event-based vision community.

OpenEB is composed of the [Metavision SDK Open Modules](https://docs.prophesee.ai/stable/modules.html#chapter-modules-and-packaging-open):
* **HAL**: Hardware Abstraction Layer to operate any event-based vision device.
* **Base**: Foundations and common definitions of event-based applications.
* **Core**: Generic algorithms for visualization, event stream manipulation, applicative pipeline generation.
* **Core ML**: Generic functions for Machine Learning, event_to_video and video_to_event pipelines.
* **Driver**: High-level abstraction built on the top of HAL to easily interact with event-based cameras.
* **UI**: Viewer and display controllers for event-based data.

OpenEB also contains the source code for [Prophesee camera plugins](https://docs.prophesee.ai/stable/installation/camera_plugins.html) which allow applications to handle I/O operations. Once enabled, applications can stream data from event-cameras, as well as read and playback event recordings.

## Supported Cameras
* EVK2 - HD
* EVK3 - VGA/320/HD
* EVK4 - HD

## Build, Test, and Install OpenEB
Follow the links below to find the build and installation guide for your OS.

- [**Linux**](./docs/build-and-installation-guides/linux-build-test-and-installation.md)
- [**MacOS**](./docs/build-and-installation-guides/macos-build-test-and-installation.md)
- [**Windows**](./docs/build-and-installation-guides/windows-build-test-and-installation.md)

## Additional Resources
- [**Metavision SDK Docs**](https://docs.prophesee.ai/stable/index.html)
- [**Get started** guides for C++ and Python](https://docs.prophesee.ai/stable/get_started/index.html)
- [**Code samples** for API best practices](https://docs.prophesee.ai/stable/samples.html)
- [**Technical details** regarding modules and packaging](https://docs.prophesee.ai/stable/modules.html)
