#!/bin/sh
echo "begin..."
python simclr_sassub.py --subset-indices ./sas_subset/cifar100gau-0.2-sas-indices.pkl
wait
python simclr_sassub.py --subset-indices ./sas_subset/cifar100gau-0.05-sas-indices.pkl
wait
python simclr_sassub.py --subset-indices ./sas_subset/cifar100gau-0.1-sas-indices.pkl
wait
python simclr_sassub.py --subset-indices ./sas_subset/cifar100gau-0.4-sas-indices.pkl
wait
python simclr_sassub.py --subset-indices ./sas_subset/cifar100gau-0.6-sas-indices.pkl
wait
python simclr_sassub.py --subset-indices ./sas_subset/cifar100gau-0.8-sas-indices.pkl
wait
python simclr_sassub.py --subset-indices ./sas_subset/cifar100gau-0.15-sas-indices.pkl
wait
echo "end..."
