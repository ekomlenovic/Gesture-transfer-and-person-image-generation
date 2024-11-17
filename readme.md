
# Objectif de ce code
(Ce code est indépendant de tout le reste du dépôt)

À partir d'une vidéo d'une personne source et d'une autre d'une personne, notre objectif est de générer une nouvelle vidéo de la cible effectuant les mêmes mouvements que la source. 

[Allez voir le sujet du TP ici](http://alexandre.meyer.pages.univ-lyon1.fr/m2-apprentissage-profond-image/am/tp_dance/)

# Comment utiliser ce code

## Requis

- Python 3.11
- environnement virtuel (conda or venv)

## Installation

```bash

conda env create -f environment.yml
conda activate image

```

## Utilisation - Entrainer le modèle

```bash

python VideoSkeleton.py # Pour generer le dataset

# Pour choisir le modele parmis Vanilla, choisir dans le fichier optSkeOrImage 1 ou 2
python GenVanillaNN.py 
# Pour le gan même principe 
python GenGAN.py

```

## Utilisation - Tester un modèle entrainé

```bash

# Dans le fichier DanceDemo.py, choisir parmis puis le lancer
# NEAREST = 1
# VANILLA_NN_SKE = 2
# VANILLA_NN_Image = 3
# GAN_SKE  = 4
# GAN_Image = 5

python DanceDemo.py

```

## Mon travail

Pour l'ensemble des modèles, j'ai essayé d'augmenter le dataset, en doublant le dataset (modulo 10 -> modulo 5), cepandant, je n'ai pas constaté de nettes améliorations. J'ai rendu les entrainements des modèles GenVanillaNN et GenGAN compatibles avec les GPU.

### 1. GenNearest.py

Recherche dans l’ensemble de données, l’image dont le squelette associé est le plus proche.

### 2. GenVanillaNN.py

Il y a deux modèles, un pour les squelettes et un pour les images. `GenNNSkeToImage`, `GenNNSkeImToImage`.
Pour `GenNNSkeImToImage`, j'ai bien transformé ma source au préalable.


### 3. GenGAN.py

J'ai implémenter le modèle GAN, avec les deux versions (skeleton et image). J'ai remarqué que le Discriminateur était très performant, mais le générateur ne l'était pas. J'ai essayé de modifier les hyperparamètres et de rendre le discriminateur moins performant. Pour cela, j'ai mis un dropout, modifié le lr, et changer les labels de 0 et 1 pour 0.1 et 0.9.


# Auteurs

- Emilien KOMLENOVIC p2000315
- [Alexandre Meyer](http://alexandre.meyer.pages.univ-lyon1.fr/) base du code 