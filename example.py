#!/usr/bin/env python3
# Copyright 2018 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from time import sleep
import numpy as np

from arriz import Arriz


def main():
    buffer_len, feature_size = 30, 16
    buffer = np.zeros((buffer_len, feature_size))
    while Arriz.show('Data', buffer):
        new_feature = buffer[-1] * 0.5 + 0.5 * np.random.random((feature_size,))
        buffer = np.concatenate([buffer[1:], [new_feature]])
        sleep(0.1)


if __name__ == '__main__':
    main()
