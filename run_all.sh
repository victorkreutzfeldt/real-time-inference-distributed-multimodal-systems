#!/bin/bash

VARIANTS=("SotA" "PaMo" "ToMo")
AUDIO_SNRS=("-5" "1.1888" "2")

for variant in "${VARIANTS[@]}"
do
  for snr in "${AUDIO_SNRS[@]}"
  do
    echo "Running variant=${variant}, audio_snr_dB=${snr}"
    python3.13 run_wrapper.py --variant "$variant" --audio_snr_dB "$snr"
  done
done