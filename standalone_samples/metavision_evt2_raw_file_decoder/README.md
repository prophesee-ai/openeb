# Metavision EVT2 RAW File decoder sample

This code sample demonstrates how to decode an EVT2 format RAW file and write the decoded events in a CSV file

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
./metavision_evt2_raw_file_decoder INPUT_FILENAME CD_CSV_OUTPUT_FILENAME (TRIGGER_CSV_OUTPUT_FILENAME)
```

The CD CSV file will have the format : x;y;polarity;timestamp
The Trigger output CSV file (if provided) will have the format : value;id;timestamp

