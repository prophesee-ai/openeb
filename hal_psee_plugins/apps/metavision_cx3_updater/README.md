# Metavision cx3 updater

This code sample demonstrates how to flash the system firmware and FPGA bitstream.

## Installation

First, make a copy of the code sample, so that you can work with this copy.
Go to the directory with your copy of the code sample and compile it.

```
mkdir build
cd build
cmake ..
cmake --build .
```

In this code sample, our CMakeLists.txt is configured in a such way that the executable is generated in the "build" directory.

## Running

Go to the "build" directory.

To start the sample 

```
./metavision_cx3_updater
```

To update the fpga bitstream

```
./metavision_cx3_updater --usb-fpga fpga.bit
```

To update the firmware

```
./metavision_cx3_updater --usb-fw firmware.bit
```