#!/bin/bash

# download model weights and extra files
gdown 1W53UMg8kee3HGRTNd2aNhMUew_kj36OH
unzip -q data.zip
rm data.zip

# download dataset annotations
gdown 1f-D3xhQPMC9rwtaCVNoxtD4BQh4oQbY9
unzip -q datasets.zip
rm datasets.zip
mv datasets/ data/
