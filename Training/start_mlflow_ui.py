#!/usr/bin/env python3
"""
Script pour démarrer l'interface web MLflow
"""
import subprocess
import sys
import os

def start_mlflow_ui():
    """
    Démarre l'interface web MLflow sur le port 5000
    """
    print("Démarrage de l'interface web MLflow...")
    print("L'interface sera accessible à l'adresse: http://localhost:5000")
    print("Appuyez sur Ctrl+C pour arrêter le serveur")
    
    try:
        # Changer vers le répertoire Training pour accéder aux mlruns
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Démarrer MLflow UI
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui", 
            "--host", "0.0.0.0", 
            "--port", "5000"
        ])
    except KeyboardInterrupt:
        print("\nArrêt de l'interface MLflow.")
    except Exception as e:
        print(f"Erreur lors du démarrage de MLflow: {e}")

if __name__ == "__main__":
    start_mlflow_ui()
