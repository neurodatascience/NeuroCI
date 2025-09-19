#!/usr/bin/env bash
set -euo pipefail

# Arguments
LICENSE_FILE=$1
INPUTS=$2
OUTPUT_DIR=$3
THREADS=$4
PALLIDUM_SEPARATE=$5
STD_SPACE=$6

# Export FreeSurfer license
export FS_LICENSE="${LICENSE_FILE}"

echo "=== Running SAMSEG ==="
run_samseg \
  --input "${INPUTS}" \
  --output "${OUTPUT_DIR}" \
  "${THREADS}" \
  "${PALLIDUM_SEPARATE}"

echo "=== Converting to NIfTI ==="
mri_convert \
  "${OUTPUT_DIR}/seg.mgz" \
  "${OUTPUT_DIR}/seg.nii.gz"

echo "=== Running antsRegistration on T1 ==="
antsRegistration \
  --dimensionality 3 \
  --float 0 \
  --output "[${OUTPUT_DIR}/T1_MNI_,${OUTPUT_DIR}/T1_MNI_Warped.nii.gz]" \
  --winsorize-image-intensities [0.005,0.995] \
  --use-histogram-matching 1 \
  --initial-moving-transform "[${STD_SPACE},${INPUTS},1]" \
  --transform Rigid[0.1] \
  --metric "MI[${STD_SPACE},${INPUTS},1,32,Regular,0.25]" \
  --convergence [1000x500x250x100,1e-6,10] \
  --shrink-factors 8x4x2x1 \
  --smoothing-sigmas 3x2x1x0vox \
  --transform Affine[0.1] \
  --metric "MI[${STD_SPACE},${INPUTS},1,32,Regular,0.25]" \
  --convergence [1000x500x250x100,1e-6,10] \
  --shrink-factors 8x4x2x1 \
  --smoothing-sigmas 3x2x1x0vox \
  --transform SyN[0.1,3,0] \
  --metric "CC[${STD_SPACE},${INPUTS},1,4]" \
  --convergence [100x70x50x20,1e-6,10] \
  --shrink-factors 8x4x2x1 \
  --smoothing-sigmas 3x2x1x0vox \
  -v

echo "=== Applying transform to segmentation (nearest-neighbor) ==="
antsApplyTransforms \
  -d 3 \
  -i "${OUTPUT_DIR}/seg.nii.gz" \
  -r "${STD_SPACE}" \
  -o "${OUTPUT_DIR}/seg_MNI.nii.gz" \
  -n NearestNeighbor \
  -t "${OUTPUT_DIR}/T1_MNI_1Warp.nii.gz" \
  -t "${OUTPUT_DIR}/T1_MNI_0GenericAffine.mat"

echo "=== Done ==="

