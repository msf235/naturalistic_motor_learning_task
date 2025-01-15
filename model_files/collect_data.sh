#!/usr/bin/env bash

# Directory where MJDATA.txt is created/exists
DIR="./"
cd "$DIR" || exit 1

# Check if the base file exists
if [[ -f "MJDATA.TXT" ]]; then
  # Find the highest existing index among MJDATA_*.txt
  # 1. ls MJDATA_*.txt 2>/dev/null        => list existing indexed files (suppress error if none)
  # 2. sed -n 's/.*_\([0-9]\+\)\.txt/\1/p' => extract only the numeric index
  # 3. sort -n | tail -1                  => get the highest numeric index
  last_index=$(
    ls MJDATA_*.txt 2>/dev/null |
      sed -n 's/.*_\([0-9]\+\)\.txt/\1/p' |
      sort -n |
      tail -1
  )

  # If no indexed file exists, set last_index to -1
  [[ -z "$last_index" ]] && last_index=-1

  # The new file index will be last_index + 1
  new_index=$((last_index + 1))

  # Rename MJDATA.txt to MJDATA_<new_index>.txt
  mv "MJDATA.TXT" "MJDATA_${new_index}.txt"
  echo "Renamed MJDATA.TXT to MJDATA_${new_index}.txt"
else
  echo "No MJDATA.TXT file found."
fi
