#!/bin/bash

AUDIO_SNRS=("-5" "1.1888" "2")

for snr in "${AUDIO_SNRS[@]}"
do
  echo "Plotting audio_snr_dB=${snr}"
  python3.13 plotting_per_audio_snr.py --audio_snr_dB "$snr"
done
