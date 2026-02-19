#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <estimator> <robot_number> <trial> <output_directory (optional)>"
    exit 1
fi

export OMP_NUM_THREADS=8
sudo cpupower frequency-set -g performance

if [ "$#" -ge 4 ]; then
    mkdir -p "$4"
    nice -n -20 ./build/release/estimator "assets/config/estimator/$1.json"  "assets/config/robot/robot$2.json" "assets/config/trial/$3.json" --output_path "$4/"
else
    nice -n -20 ./build/release/estimator "assets/config/estimator/$1.json"  "assets/config/robot/robot$2.json" "assets/config/trial/$3.json"
fi

sudo cpupower frequency-set -g powersave