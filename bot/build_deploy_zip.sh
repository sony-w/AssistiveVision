#!/bin/bash

# create a file to upload lambda code
rm deploy.zip
cd ./src
zip ../deploy.zip *
cd ..
aws lambda update-function-code \
    --function-name arn:aws:lambda:us-east-2:387946532044:function:webhook_post \
    --zip-file fileb://deploy.zip