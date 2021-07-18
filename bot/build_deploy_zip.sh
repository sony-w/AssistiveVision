#!/bin/bash

# create a file to upload lambda code
rm deploy.zip
cd ./src
zip ../deploy.zip *
cd ..