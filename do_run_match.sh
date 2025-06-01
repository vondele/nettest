#!/bin/bash

NETDIR=$1
MATCHDIR=$2
ROUNDS=$3

cd $NETDIR
networks=$(echo nn-*.nnue)

ENGINES=""
for network in $networds
do
    ENGINES="$ENGINES -engine name=$network cmd=/workspace/Stockfish/src/stockfish option.EvalFile=$NETDIR/$network "
done

mkdir -p $MATCHDIR
cd $MATCHDIR

/workspace/fastchess/fastchess -rounds $ROUNDS -games 2 -repeat -srand 42  -concurrency 280 \
                               -openings file=/workspace/data/UHO_Lichess_4852_v1.epd format=epd order=random \
                               -each proto=uci option.Threads=1 option.Hash=16 tc=10+0.1 \
                               -ratinginterval 280 -report penta=true -pgnout file=match.pgn \
                               -engine name=master cmd=/workspace/Stockfish/src/stockfish \
                               $ENGINES \
                               | tee fastchess.out | grep -v "Started game" | grep -v "Started game"
