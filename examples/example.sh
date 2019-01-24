#!/bin/bash

cd ../build/
./bin/mumfordShah -i intensity.png -m mask.png -o result.png -a -1 -l 0.1 -s shading.png
