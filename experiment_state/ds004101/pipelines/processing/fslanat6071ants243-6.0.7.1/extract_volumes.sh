#!/bin/bash

set -euo pipefail
echo "Initiating script..."

# === Parse Inputs ===
t1="$1"             # raw T1 input
anat_outdir="$2"    # output root
std_space="$3"      # user-supplied MNI template (NIfTI)

# === Run fsl_anat ===
echo "Running fsl_anat..."
fsl_anat -i "$t1" -o "$anat_outdir"

# === Paths (fsl_anat outputs) ===
anat_base="${anat_outdir}.anat"
seg="${anat_base}/T1_subcort_seg.nii.gz"
outfile="${anat_base}/subcortical_volumes.json"

# === Extract Volumes ===
echo "{" > "$outfile"
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
echo "}" >> "$outfile"
echo "✅ Subcortical volumes saved to $outfile"

# === Registration with ANTs ===
reg_prefix="${anat_base}/T1_to_MNI_"
t1_warped="${reg_prefix}Warped.nii.gz"
seg_mni="${anat_base}/subcortical_seg_MNI_ANTs.nii.gz"

echo "=== Running antsRegistration (T1 -> MNI) ==="
antsRegistration \
  --dimensionality 3 \
  --float 0 \
  --output "[${reg_prefix},${t1_warped}]" \
  --winsorize-image-intensities [0.005,0.995] \
  --use-histogram-matching 1 \
  --initial-moving-transform "[${std_space},${t1},1]" \
  --transform Rigid[0.1] \
  --metric "MI[${std_space},${t1},1,32,Regular,0.25]" \
  --convergence [1000x500x250x100,1e-6,10] \
  --shrink-factors 8x4x2x1 \
  --smoothing-sigmas 3x2x1x0vox \
  --transform Affine[0.1] \
  --metric "MI[${std_space},${t1},1,32,Regular,0.25]" \
  --convergence [1000x500x250x100,1e-6,10] \
  --shrink-factors 8x4x2x1 \
  --smoothing-sigmas 3x2x1x0vox \
  --transform SyN[0.1,3,0] \
  --metric "CC[${std_space},${t1},1,4]" \
  --convergence [100x70x50x20,1e-6,10] \
  --shrink-factors 8x4x2x1 \
  --smoothing-sigmas 3x2x1x0vox \
  -v

echo "=== Applying transforms to segmentation (nearest-neighbor) ==="
antsApplyTransforms \
  -d 3 \
  -i "$seg" \
  -r "$std_space" \
  -o "$seg_mni" \
  -n NearestNeighbor \
  -t "${reg_prefix}1Warp.nii.gz" \
  -t "${reg_prefix}0GenericAffine.mat"

echo "=== Done ==="
echo "Warped segmentation: $seg_mni"
echo "Warped T1 (ANTs): $t1_warped"
echo "Transforms: ${reg_prefix}0GenericAffine.mat, ${reg_prefix}1Warp.nii.gz"

