🚀 DL_Engineering_TP1 : MLP MNIST et MLOps🌟 

Ce dépôt contient l'implémentation du Travail Pratique 1 (TP 1) du cours de Deep Learning Engineering. 
L'objectif est de développer, suivre et déployer un modèle de classification d'images basé sur un Réseau de Neurones Multi-Couches (MLP) pour le jeu de données MNIST (chiffres manuscrits). 
Le projet met en pratique les étapes clés du cycle de vie des modèles (MLOps) :

Entraînement (TensorFlow/Keras).

Suivi d'Expérimentations (MLflow).

Conteneurisation.

Déploiement via une API Flask et Docker.




🛠️ Prise en Main et Installation
1. Prérequis : Assurez-vous d'avoir installé :
Python 3.8+; Docker Desktop (pour le déploiement) et Git.
2. Cloner le Dépôt
git clone <URL_DE_VOTRE_DEPOT>
cd DL_Engineering_TP1
4. Environnement Virtuel et Dépendances
Créez et activez l'environnement virtuel, puis installez les dépendances listées dans requirements.txt.
# Création et activation de l'environnement
python -m venv .venv

# Activer l'environnement (PowerShell Windows)
.\.venv\Scripts\Activate.ps1
# OU (Linux/macOS)
source .venv/bin/activate

# Installation des dépendances
pip install -r requirements.txt
🏃 Utilisation : Entraînement et Suivi MLflow

Étape 1 : Entraînement du Modèle
Le script entraîne le modèle MLP, effectue l'évaluation, et procède au suivi MLflow.Bashpython train_model.py
Sortie attendue : Le script affiche la précision finale et indique que le modèle est enregistré dans le registre de modèles MLflow sous le nom MNIST_MLP_Model.

Étape 2 : Lancement de l'Interface MLflow Pour analyser en détail la convergence (Loss et Accuracy) et les hyperparamètres (Époques=5, Batch_Size=128, Dropout=0.2) :
python -m mlflow ui
Accédez à l'interface dans votre navigateur : http://127.0.0.1:5000
