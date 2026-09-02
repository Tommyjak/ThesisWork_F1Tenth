#!/bin/bash

for v in ../../datasets/videos/test_camera_*.mp4; do
	./run.sh "$v"
done
