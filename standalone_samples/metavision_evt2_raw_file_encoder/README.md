# Metavision EVT2 RAW File encoder sample

This code sample demonstrates how to encode events in a CSV file into an EVT2 format RAW file

## Installation

First, make a copy of the code sample, so that you can work with this copy.
Go to the directory with your copy of the code sample and compile it.

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

In this code sample, our CMakeLists.txt is configured in a such way that the executable is generated in the "build" directory.


## Running

Go to the "build" directory.


```
./metavision_evt2_raw_file_encoder OUTPUT_FILENAME CD_CSV_INPUT_FILENAME (TRIGGER_CSV_INPUT_FILENAME)
```

The CD CSV file needs to have the format : x;y;polarity;timestamp
The Trigger CSV file (if provided) needs to have the format : value;id;timestamp

