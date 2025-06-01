#!/bin/bash

export BASE=$1
export OWNER=$2
export REPO=$3

export GIT_LFS_CONCURRENT_TRANSFERS=8

mkdir -p $BASE/$OWNER
cd $BASE/$OWNER

if [ -f $REPO/.git/config ]; then
  cd $REPO
  git pull
else
  rm -f $REPO
  git clone https://huggingface.co/datasets/$OWNER/$REPO
fi
