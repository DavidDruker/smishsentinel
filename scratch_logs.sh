#!/bin/bash
export AWS_SHARED_CREDENTIALS_FILE=/mnt/c/Users/DavidNew/.aws/credentials
export AWS_CONFIG_FILE=/mnt/c/Users/DavidNew/.aws/config
export AWS_PROFILE=agentsforhumans
cd /mnt/c/Users/DavidNew/Documents/smishsentinel
aws logs tail /aws/bedrock-agentcore/runtimes/app-9IoGY77lPi-DEFAULT \
  --log-stream-name-prefix "2026/08/26/[runtime-logs]" --since 10m --format short \
  > scratch_logwindow.txt 2>&1
wc -l scratch_logwindow.txt
