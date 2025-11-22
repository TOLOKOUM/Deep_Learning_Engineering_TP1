import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.tensorflow

# --- Définition des Paramètres de l'Expérience (Exercice 2) ---
EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2
MLFLOW_RUN_NAME = "mlp_mnist_initial_run"

# --- Chargement et Préparation des Données (Exercice 1) ---
print("--- 1. Chargement du jeu de données MNIST ---")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisation des données [cite: 1111, 1112]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Mise en forme (Flatten) des images (28x28) en un vecteur (784) [cite: 1043]
# Nécessaire pour un réseau de neurones fully-connected (MLP)
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# --- Lancement de la Session de Suivi MLflow (Exercice 2) ---
mlflow.set_experiment("MNIST_MLP_Classification")
with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as run:
    print("--- 2. Suivi MLflow démarré ---")
    
    # Enregistrement des paramètres [cite: 1062]
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)
    mlflow.log_param("optimizer", "adam")

    # --- Construction du Modèle MLP (Exercice 1) ---
    print("--- 3. Construction du Modèle ---")
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)), # Couche Dense [cite: 1043]
        keras.layers.Dropout(DROPOUT_RATE), # Couche Dropout [cite: 1043]
        keras.layers.Dense(10, activation='softmax') # Couche de sortie (10 classes) [cite: 1043]
    ])

    # Compilation du modèle [cite: 1043]
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', # Perte adaptée pour les labels entiers
        metrics=['accuracy']
    )

    # --- Entraînement du Modèle (Exercice 1) ---
    print(f"--- 4. Entraînement du Modèle sur {EPOCHS} époques ---")
    mlflow.tensorflow.autolog(log_models=False) # Permet à MLflow d'enregistrer automatiquement les métriques
    
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1 # 10% des données d'entraînement pour la validation [cite: 1043]
    )

    # --- Évaluation du Modèle (Exercice 1) ---
    print("--- 5. Évaluation du Modèle ---")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Précision sur l'ensemble de test: {test_acc:.4f}")

    # Enregistrement manuel de la métrique finale sur l'ensemble de test (Exercice 2)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_loss", test_loss)

    # --- Sauvegarde du Modèle (Exercice 3) ---
    # Sauvegarde du modèle dans le format Keras natif
    model_path = "model_artefact.h5"
    # ... (lignes précédentes)

    # Enregistrement manuel de la métrique finale sur l'ensemble de test (Exercice 2)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_loss", test_loss)

    # --- Sauvegarde du Modèle (Exercice 3) ---
    # Sauvegarde du modèle dans le format Keras natif
    model_path_local = "model_artefact.h5" # NOM DU FICHIER LOCAL (avec extension)
    model.save(model_path_local)
    
    # Enregistrement du modèle complet dans MLflow comme un artefact
    print("--- 6. Sauvegarde et Enregistrement du Modèle via MLflow ---")
    mlflow.keras.log_model(
        model=model, 
        # artifact_path DOIT être un nom de dossier SANS extension ni caractères spéciaux
        artifact_path="model_artefact", 
        registered_model_name="MNIST_MLP_Model" # Le nom du modèle enregistré, sans extension
    )
    
    # Affichage de l'URI pour l'accès
    model_uri = mlflow.get_artifact_uri("model_artefact")
    print(f"\nModèle enregistré dans : {model_uri}")
    print(f"Pour accéder à l'interface MLflow, exécutez 'mlflow ui' dans le terminal.")
    # Affichage de l'URI pour l'accès
    model_uri = mlflow.get_artifact_uri(model_path)
    print(f"\nModèle enregistré dans : {model_uri}")
    print(f"Pour accéder à l'interface MLflow, exécutez 'mlflow ui' dans le terminal.")