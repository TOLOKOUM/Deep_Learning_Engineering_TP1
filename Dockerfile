# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier des dépendances et les installer
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers nécessaires pour l'API et le modèle (mlruns, app.py)
COPY . .

# Exposer le port de l'application Flask
EXPOSE 5000

# Commande pour démarrer l'application
CMD ["python", "app.py"]