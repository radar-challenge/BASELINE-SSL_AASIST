#!/bin/bash

DATANAME=LlamaPartialSpoof-full
LABELFILE=label_LlamaPartialSpoof-full.txt
SCPFILE=$DATANAME/wav_full.scp

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

./download_lps.sh
mv LlamaPartialSpoof LlamaPartialSpoof-full

# Prepare wav list in the same "<utt_id> <path>" format as flac.scp
echo "[INFO] Preparing wav list: $SCPFILE"
awk -v base="$DATANAME/R01TTS.0.a" '{print $1 " " base "/" $1 ".wav"}' "$LABELFILE" > "$SCPFILE"

# Run inference
echo "[INFO] Running inference..."
python inference.py \
  --ssl_cp_path "checkpoints/xlsr2_300m_fixed.pt" \
  --checkpoint_path $MODELCKPT \
  --wav_list $SCPFILE \
  --output_path $DATANAME/scores.txt \
  --batch_size 8 \
  --num_workers 2 \
  --device $DEVICE
