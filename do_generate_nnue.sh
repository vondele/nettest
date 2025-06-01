#!/bin/bash

set -e

ROOTDIR=$1
TARGET=$2


LASTCKPT=$ROOTDIR/lightning_logs/version_0/checkpoints/last.ckpt
LASTNNUE=$ROOTDIR/lightning_logs/version_0/checkpoints/last.nnue

FTOPTDATA=/workspace/data/official-stockfish/master-binpacks/test80-2022-08-aug-16tb7p.v6-dd.min.binpack


# TODO: --ft_optimize (too slow for testing)

export OMP_NUM_THREADS=8
python serialize.py $LASTCKPT $LASTNNUE --features=HalfKAv2_hm^ --ft_compression=leb128 --ft_optimize_count=1000000 --ft_optimize_data=$FTOPTDATA

NNUESHA=$(sha256sum $LASTNNUE | cut -c1-12)
echo "copying $LASTNNUE to nn-$NNUESHA.nnue"
mkdir -p $TARGET
cp $LASTNNUE TARGET/nn-$NNUESHA.nnue

