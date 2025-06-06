#!/bin/bash

set -e

NETDIR=$1
MATCHDIR=$2
ROUNDS=$3

echo "Net dir: $NETDIR"
cd $NETDIR
networks=$(echo nn-*.nnue)

echo "Found networks: " $networks

ENGINES=""
for network in $networks
do
    ENGINES="$ENGINES -engine name=$network cmd=/workspace/Stockfish/src/stockfish option.EvalFile=$NETDIR/$network "
done

echo "Match dir: $MATCHDIR"
mkdir -p $MATCHDIR
cd $MATCHDIR

/workspace/fastchess/fastchess -rounds $ROUNDS -games 2 -repeat -srand 42  -concurrency 280 \
                               -openings file=/workspace/data/UHO_Lichess_4852_v1.epd format=epd order=random \
                               -ratinginterval 280 -report penta=true -pgnout file=match.pgn \
                               -engine name=master cmd=/workspace/Stockfish/src/stockfish \
                               $ENGINES \
                               -each proto=uci option.Threads=1 option.Hash=16 tc=10+0.1 \
                               | tee fastchess.out | grep -v "Started game" | grep -v "Finished game"
