#!/bin/bash

DATANAME=RADAR2026-dev

DEVICE=cuda
MODELCKPT=checkpoints/Best_LA_model_for_DF.pth

echo "[INFO] Starting run for dataset: $DATANAME"

# Create fixed SSL checkpoint if needed
if [ ! -f "checkpoints/xlsr2_300m_fixed.pt" ] && [ -f "checkpoints/xlsr2_300m.pt" ]; then
  echo "[INFO] Creating fixed SSL checkpoint: checkpoints/xlsr2_300m_fixed.pt"
  python fix_xlsr2.py checkpoints/xlsr2_300m.pt "checkpoints/xlsr2_300m_fixed.pt"
else
  echo "[INFO] Using existing fixed SSL checkpoint (checkpoints/xlsr2_300m_fixed.pt)."
fi


# Prepare scp file
echo "[INFO] Preparing wav list: $DATANAME/flac.scp"
find "$DATANAME/flac/" -type f -name "*.flac" | sort | awk -F/ '{id=$NF; sub(/\.flac$/, "", id); print id " " $0}' > "$DATANAME/flac.scp"


# Run inference
echo "[INFO] Running inference..."
python inference.py \
  --ssl_cp_path "checkpoints/xlsr2_300m_fixed.pt" \
  --checkpoint_path $MODELCKPT \
  --wav_list $DATANAME/flac.scp \
  --output_path $DATANAME/scores.txt \
  --batch_size 8 \
  --num_workers 2 \
  --device $DEVICE

echo "[INFO] Writing sorted score file: submissions/$DATANAME/score.tsv"
mkdir -p submissions/$DATANAME/
echo -e "filename\tscore" > submissions/$DATANAME/score.tsv
awk '{print $1"\t"$2}' $DATANAME/scores.txt | sort >> submissions/$DATANAME/score.tsv
cd submissions/$DATANAME/
zip submission.zip score.tsv
cd ../../
echo "[INFO] Done. Please submit submissions/$DATANAME/submission.zip to the APSIPA RADAR challenge 2026"