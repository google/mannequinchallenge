#!/bin/bash
cd test_data
curl -L https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip -o DAVIS-data.zip
unzip DAVIS-data.zip
rm DAVIS-data.zip
cd ..
