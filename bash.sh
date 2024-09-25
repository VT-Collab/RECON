#!/bin/bash

for i in {1.15}
do
    python get_data.py --scenarios 10
    python train_vision_model.py
    python train_vision_model.py --beacon "Exact"
    python train_vision_model.py --beacon "Partial"
    python train_vision_model.py --beacon "Other"
    python train_vision_model.py --beacon "Random"
done