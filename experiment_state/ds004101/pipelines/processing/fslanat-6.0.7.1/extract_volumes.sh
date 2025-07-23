#!/bin/bash

# Exit on error
set -e

# === Parse Inputs ===
t1="$1"
anat_outdir="$2"

if [[ -z "$t1" || -z "$anat_outdir" ]]; then
  echo "Usage: $0 <T1_IMAGE> <OUTPUT_DIR>"
  exit 1
fi

# === Run fsl_anat ===
echo "Running fsl_anat..."
fsl_anat -i "$t1" -o "$anat_outdir"

# === Define Segmentation Path ===
seg="${anat_outdir}.anat/T1_subcort_seg.nii.gz"
outfile="${anat_outdir}.anat/subcortical_volumes.json"

# === Start JSON File ===
echo "{" > "$outfile"

# === Define Labels and Thresholds ===
labels=(
  "Left-Thalamus:9.5:10.5"
  "Left-Caudate:10.5:11.5"
  "Left-Putamen:11.5:12.5"
  "Left-Pallidum:12.5:13.5"
  "Brainstem:15.5:16.5"
  "Left-Hippocampus:16.5:17.5"
  "Left-Amygdala:17.5:18.5"
  "Left-Accumbens-area:25.5:26.5"
  "Right-Thalamus:48.5:49.5"
  "Right-Caudate:49.5:50.5"
  "Right-Putamen:50.5:51.5"
  "Right-Pallidum:51.5:52.5"
  "Right-Hippocampus:52.5:53.5"
  "Right-Amygdala:53.5:54.5"
  "Right-Accumbens-area:57.5:58.5"
)

# === Extract Volumes ===
for i in "${!labels[@]}"; do
  IFS=":" read -r name l u <<< "${labels[$i]}"
  stats=$(fslstats "$seg" -l "$l" -u "$u" -V)
  voxels=$(echo "$stats" | awk '{print $1}')
  volume=$(echo "$stats" | awk '{print $2}')

  echo "$name: $voxels voxels, $volume mm³"

  if [[ $i -lt $((${#labels[@]} - 1)) ]]; then
    echo "  \"$name\": $volume," >> "$outfile"
  else
    echo "  \"$name\": $volume" >> "$outfile"
  fi
done

# === Finalize JSON ===
echo "}" >> "$outfile"

echo "✅ Subcortical volumes saved to $outfile"

