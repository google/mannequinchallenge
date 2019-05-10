#!/bin/bash
cd test_data
wget https://storage.googleapis.com/mannequinchallenge-data/tumtest/tum_hdf5.tgz
tar -xvzf tum_hdf5.tgz
rm tum_hdf5.tgz
cd ..
