#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <DATANAME> <LABELFILE>"
  exit 1
fi

DATANAME="$1"
LABELFILE="$2"

MINVAL=-8
MAXVAL=8
RESOLUTION=2000
SCOREINDEX=1
THRESHOLD=0.0

resultdir="result_$DATANAME"


mkdir -p "$resultdir"

echo "$0: calculate utterance-based EER"
python partialspoof-metrics/calculate_eer.py --labpath "${LABELFILE}" \
                          --scopath "${DATANAME}/scores.txt" \
                          --savepath "$resultdir/"\
                          --resolution $RESOLUTION \
                          --scoreindex $SCOREINDEX \
                          --minval "$MINVAL" \
                          --maxval "$MAXVAL"

echo "$0: Draw score distribution dentisy figure and save at ${resultdir}/score.pdf"
python partialspoof-metrics/draw_score_distribution.py --loadpath "${resultdir}" \
                                    --savepath "${resultdir}/score.pdf" \
                                    --threshold $THRESHOLD \
                                    --xmin "$MINVAL" \
                                    --xmax "$MAXVAL"