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

mkdir -p checkpoints/test_local
curl -L https://storage.googleapis.com/mannequinchallenge-data/checkpoints/best_depth_Ours_Bilinear_inc_3_net_G.pth -o checkpoints/test_local/best_depth_Ours_Bilinear_inc_3_net_G.pth
curl -L https://storage.googleapis.com/mannequinchallenge-data/checkpoints/best_depth_Ours_Bilinear_inc_6_net_G.pth -o checkpoints/test_local/best_depth_Ours_Bilinear_inc_6_net_G.pth
curl -L https://storage.googleapis.com/mannequinchallenge-data/checkpoints/best_depth_Ours_Bilinear_inc_7_net_G.pth -o checkpoints/test_local/best_depth_Ours_Bilinear_inc_7_net_G.pth

