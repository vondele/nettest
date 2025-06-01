#!/bin/bash

set -e

export BASE=$1
export OWNER=$2
export REPO=$3

starttime=$(date +%s)

export GIT_LFS_CONCURRENT_TRANSFERS=8

mkdir -p $BASE/$OWNER
cd $BASE/$OWNER

if [ -f $REPO/.git/config ]; then
  cd $REPO
  git pull
else
  rm -f $REPO
  git clone https://huggingface.co/datasets/$OWNER/$REPO
  cd $REPO
fi

endtime=$(date +%s)

echo "Total repo size: " $(du -s . | awk '{print int($1/(1024*1024)), "GB"}') " updated in " $((endtime-starttime)) " seconds." 
