#!/bin/bash

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mkdir -p test_data
cd test_data
curl -L https://storage.googleapis.com/mannequinchallenge-data/tumtest/tum_hdf5.tgz -o tum_hdf5.tgz
tar -xvzf tum_hdf5.tgz
rm tum_hdf5.tgz
cd ..
