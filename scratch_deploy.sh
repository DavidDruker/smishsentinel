#!/bin/bash
export AWS_SHARED_CREDENTIALS_FILE=/mnt/c/Users/DavidNew/.aws/credentials
export AWS_CONFIG_FILE=/mnt/c/Users/DavidNew/.aws/config
export AWS_PROFILE=agentsforhumans
export AGENTCORE_SUPPRESS_RECOMMENDATION=1
cd /mnt/c/Users/DavidNew/Documents/smishsentinel
./.venv-wsl/bin/agentcore deploy
