import torch
import pickle
import joblib
from pathlib import Path
import json
from datetime import datetime

def save_active_learning_model(learner, metrics_dict, model_name="active_learner"):
    """
    Sauvegarde complète d'un modèle d'apprentissage actif
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path("models") / f"{model_name}_{timestamp}"
    base_path.mkdir(parents=True, exist_ok=True)

    # 1. Sauvegarder le modèle PyTorch seul (.pth)
    pytorch_model_path = base_path / "pytorch_model.pth"
    torch.save({
        'model_state_dict': learner.estimator.module_.state_dict(),
        'model_architecture': str(learner.estimator.module_),
        'device': str(learner.estimator.device),
        'training_size': len(learner.X_training),
    }, pytorch_model_path)


    # 3. Sauvegarder les métriques et métadonnées
    metadata = {
        'timestamp': timestamp,
        'training_samples': len(learner.X_training),
        'query_strategy': str(type(learner.query_strategy).__name__),
        'model_type': str(type(learner.estimator.module_).__name__),
        'device': str(learner.estimator.device),
        'metrics': metrics_dict
    }

    metadata_path = base_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # (Suppression de la sauvegarde des données d'entraînement pour alléger le modèle)

    print(f"Modèle sauvegardé dans: {base_path}")
    return str(base_path)