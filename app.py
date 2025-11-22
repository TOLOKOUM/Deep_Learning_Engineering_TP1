import os
import numpy as np
import mlflow.keras
from flask import Flask, request, jsonify
from tensorflow import keras

# --- Configuration et Chargement du Modèle (Exercice 4) ---
# Le nom du modèle enregistré dans train_model.py
REGISTERED_MODEL_NAME = "MNIST_MLP_Model"
# Charge la dernière version du modèle depuis le MLflow Model Registry
print(f"Chargement de la dernière version du modèle: {REGISTERED_MODEL_NAME}...")

try:
    # Récupérer le modèle via MLflow Model Registry [cite: 1057, 1058]
    # Ceci nécessite que vous ayez exécuté 'train_model.py' au moins une fois
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
    model = mlflow.keras.load_model(model_uri)
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    # En cas d'échec (e.g. MLflow non initialisé), utiliser un chemin local de secours
    # Cela suppose que le répertoire 'mlruns' et l'artefact existent
    try:
        model = keras.models.load_model('model_artefact')
        print("Modèle chargé depuis le chemin local de secours.")
    except Exception as local_e:
        print(f"Échec du chargement local : {local_e}. L'API ne pourra pas prédire.")
        model = None

# Initialisation de l'application Flask [cite: 1056]
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction."""
    if model is None:
        return jsonify({"error": "Modèle non chargé."}), 500
        
    try:
        # Récupération des données du corps de la requête
        data = request.get_json(force=True)
        # On s'attend à recevoir une liste/un tableau de 784 flottants normalisés [cite: 1057]
        input_data = np.array(data['input']).reshape(1, 784).astype("float32")

        # Prédiction [cite: 1057]
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        # Retourner le résultat [cite: 1057, 1058]
        return jsonify({
            'prediction': int(predicted_class),
            'probabilities': predictions[0].tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def health_check():
    """Endpoint simple pour vérifier que l'API est en ligne."""
    return "API de Classification MNIST en ligne."

# Point de départ de l'application Flask [cite: 1076]
if __name__ == '__main__':
    # Utilisation de Gunicorn pour un usage en production (recommandé par l'exercice)
    # Dans un environnement conteneurisé, le CMD du Dockerfile prend le relais
    # Pour un test local simple :
    # app.run(host='0.0.0.0', port=5000, debug=True)
    pass