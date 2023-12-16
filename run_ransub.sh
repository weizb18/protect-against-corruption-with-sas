#!/bin/sh
echo "begin..."
python simclr_ransub.py --random-subset --subset-fraction 0.2
wait
python simclr_ransub.py --random-subset --subset-fraction 0.05
wait
python simclr_ransub.py --random-subset --subset-fraction 0.1
wait
python simclr_ransub.py --random-subset --subset-fraction 0.4
wait
python simclr_ransub.py --random-subset --subset-fraction 0.6
wait
python simclr_ransub.py --random-subset --subset-fraction 0.8
wait
python simclr_ransub.py --random-subset --subset-fraction 0.15
wait
echo "end..."