For generating mmloc training, validation and test data, four files need to be execute. Put all rawdata(xml) in rawdata folder and take following steps:
1. Running sensordata_processing.py
2. Running wifidata_processing.py
3. Running combine_sensor_wifi.cpp
4. Running mmlocdata_processing.py, this will generate .npy files used in mmloc module(this step has finished and data contained in repo).