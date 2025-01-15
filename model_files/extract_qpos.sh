#!/usr/bin/env bash

# Usage:
#   ./extract_qpos.sh input_file.txt output_qpos.txt
#
# This script scans 'input_file.txt' for a section beginning with the line "QPOS"
# and continues reading numbers until the next section header (like "QVEL") or
# until a blank line. The numeric values found are written line by line to
# 'output_qpos.txt', which can then be loaded by NumPy (e.g. np.loadtxt).

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input_file> <output_file>"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Use awk to:
#  - Turn on a 'flag' when we see the line "QPOS"
#  - Turn off the flag when we hit the line "QVEL", or a blank line, or
#    another known section start
#  - While the flag is on, for each non-blank line, print the first column
#    (the numeric value) to output.
awk '
  /^QPOS/ {flag=1; next}
  /^QVEL/ || /^QPOS/ || /^[[:space:]]*$/ {flag=0}
  flag {
    # Only print if the line is not blank and has at least one numeric field
    if (NF > 0) {
      print $1
    }
  }
' "$INPUT_FILE" >"$OUTPUT_FILE"

echo "Extracted QPOS values to '$OUTPUT_FILE'."
