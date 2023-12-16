# protect-against-corruption-with-sas
Our project for UCLA CS 260D in 2023fall is **Protecting against Corruption and Poisoning with SAS
in Contrastive Learning**. This repo implements the protection against corruption with SAS, which is one part of our project.

### Corrupted Datasets
Download CIFAR100 and place it in ```data/cifar100-original/```.
Run ```data/cifar100/cifar-100-python/gauss.py``` to generate the corrupted CIFAR100 dataset and place it in ```data/cifar100/cifar-100-python/```.

### Subset Selection
Run ```cifar100gau_subset_creation_new.py``` to generate the subset with different fractions. We have selected some subsets with different fractions and saved them in the ```sas_subset``` directory.

### Contrastive Learning on the Subsets
Run ```simclr_sassub.py``` or ```run_sassub.sh``` to do the contrastive learning on subsets selected by SAS. Run ```simclr_ransub.py``` or ```run_ransub.sh``` to do the contrastive learning on randomly selected subsets.
