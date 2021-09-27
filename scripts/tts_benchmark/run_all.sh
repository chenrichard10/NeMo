#!/usr/bin/env bash

#MODELS="mixer-tts-1 mixer-tts-2 mixer-tts-3 fastpitch"
MODELS="fastpitch"
#BATCH_SIZES="1 2 4 8 16 32 64 128 256 512 1024"
BATCH_SIZES="1024"
#N_CHARSS="64 128 256 512 1024 2048 4096"
N_CHARSS="4096"
OUTPUT=${1:-"output.csv"}

# Header
result_csv="batch_size,n_chars"
for MODEL in $MODELS ; do
  result_csv="$result_csv,$MODEL"
done
result_csv="$result_csv\n"

for BATCH_SIZE in $BATCH_SIZES ; do
  for N_CHARS in $N_CHARSS ; do
    result_csv="$result_csv$BATCH_SIZE,$N_CHARS"
    for MODEL in $MODELS ; do
      set -o pipefail
      rtf=$(BATCH_SIZE=$BATCH_SIZE N_CHARS=$N_CHARS scripts/tts_benchmark/run.sh "$MODEL" | grep avg_RTF | cut -c 10-)
      # shellcheck disable=SC2181
      if [ $? -ne 0 ];
      then
        result_csv="$result_csv,NaN"
      else
        result_csv="$result_csv,$rtf"
      fi
    done
    result_csv="$result_csv\n"
  done
done

echo -ne "$result_csv" >"$OUTPUT"