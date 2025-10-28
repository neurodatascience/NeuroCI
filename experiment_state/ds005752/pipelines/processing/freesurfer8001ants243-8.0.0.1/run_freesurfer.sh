#!/usr/bin/env bash
set -euo pipefail

# Arguments:
# 1) LICENSE_FILE
# 2) INPUT (T1 NIfTI)
# 3) SUBJECTS_DIR (output root)
# 4) SUBJID
# 5) DIRECTIVES (e.g., -all)
# 6) STD_SPACE (standard reference, e.g., MNI152_T1_1mm.nii.gz)
# 7+) OPTIONAL extra flags (qcache, -mprage, -3T, etc.) -- passed through to recon-all

LICENSE_FILE=$1
INPUT=$2
SUBJECTS_DIR=$3
SUBJID=$4
DIRECTIVES=$5
STD_SPACE=$6
# capture any remaining optional flags (7..N)
EXTRA_FLAGS="${@:7}"

# Export FreeSurfer environment inside the script
export FS_LICENSE="${LICENSE_FILE}"
export SUBJECTS_DIR="${SUBJECTS_DIR}"

echo "=== FreeSurfer wrapper ==="
echo "SUBJECTS_DIR=${SUBJECTS_DIR}"
echo "SUBJID=${SUBJID}"
echo "INPUT=${INPUT}"
echo "DIRECTIVES=${DIRECTIVES}"
echo "STD_SPACE=${STD_SPACE}"
if [[ -n "${EXTRA_FLAGS}" ]]; then
  echo "Extra flags: ${EXTRA_FLAGS}"
fi

echo "=== Running recon-all ==="
recon-all -s "${SUBJID}" -sd "${SUBJECTS_DIR}" -i "${INPUT}" ${DIRECTIVES} ${EXTRA_FLAGS}

ASEG_MGZ="${SUBJECTS_DIR}/${SUBJID}/mri/aseg.mgz"
ASEG_NII="${SUBJECTS_DIR}/${SUBJID}/mri/aseg.nii.gz"

echo "=== Checking for aseg: ${ASEG_MGZ} ==="
if [[ ! -f "${ASEG_MGZ}" ]]; then
  echo "ERROR: ${ASEG_MGZ} not found. Exiting." >&2
  exit 2
fi

echo "=== Converting aseg.mgz to NIfTI ==="
mri_convert "${ASEG_MGZ}" "${ASEG_NII}"

echo "=== Running antsRegistration on raw input T1 ==="
antsRegistration \
  --dimensionality 3 \
  --float 0 \
  --output "[${SUBJECTS_DIR}/${SUBJID}/mri/T1_MNI_,${SUBJECTS_DIR}/${SUBJID}/mri/T1_MNI_Warped.nii.gz]" \
  --winsorize-image-intensities [0.005,0.995] \
  --use-histogram-matching 1 \
  --initial-moving-transform "[${STD_SPACE},${INPUT},1]" \
  --transform Rigid[0.1] \
  --metric "MI[${STD_SPACE},${INPUT},1,32,Regular,0.25]" \
  --convergence [1000x500x250x100,1e-6,10] \
  --shrink-factors 8x4x2x1 \
  --smoothing-sigmas 3x2x1x0vox \
  --transform Affine[0.1] \
  --metric "MI[${STD_SPACE},${INPUT},1,32,Regular,0.25]" \
  --convergence [1000x500x250x100,1e-6,10] \
  --shrink-factors 8x4x2x1 \
  --smoothing-sigmas 3x2x1x0vox \
  --transform SyN[0.1,3,0] \
  --metric "CC[${STD_SPACE},${INPUT},1,4]" \
  --convergence [100x70x50x20,1e-6,10] \
  --shrink-factors 8x4x2x1 \
  --smoothing-sigmas 3x2x1x0vox \
  -v

echo "=== Applying transform to aseg (nearest-neighbor) ==="
antsApplyTransforms \
  -d 3 \
  -i "${ASEG_NII}" \
  -r "${STD_SPACE}" \
  -o "${SUBJECTS_DIR}/${SUBJID}/mri/aseg_MNI.nii.gz" \
  -n NearestNeighbor \
  -t "${SUBJECTS_DIR}/${SUBJID}/mri/T1_MNI_1Warp.nii.gz" \
  -t "${SUBJECTS_DIR}/${SUBJID}/mri/T1_MNI_0GenericAffine.mat"

echo "=== Done ==="

