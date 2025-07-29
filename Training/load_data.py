# load_data.py
"""
Chargement et préparation des données pour l'apprentissage actif
"""
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

def load_images(data_dir, augment=False):
    """
    Charge toutes les images et labels depuis le dossier data_dir (ImageFolder).
    Si augment=True, applique des transformations d'augmentation de données.
    Retourne :
        - all_data (Tensor)
        - all_labels (Tensor)
        - idx_to_class (dict)
        - realwaste_dataset (ImageFolder)
    """
    if augment:
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])

    realwaste_dataset = ImageFolder(root=data_dir, transform=data_transforms)
    idx_to_class = {v: k for k, v in realwaste_dataset.class_to_idx.items()}
    
    # Afficher la distribution des classes
    print(f"Classes détectées : {realwaste_dataset.classes}")
    print(f"Nombre total d'images : {len(realwaste_dataset)}")
    
    # Compter les images par classe
    class_counts = {}
    for class_idx, class_name in idx_to_class.items():
        class_counts[class_name] = 0
    
    for _, label in realwaste_dataset.samples:
        class_name = idx_to_class[label]
        class_counts[class_name] += 1
    
    print("Distribution des images par classe :")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images")
    
    all_data, all_labels = zip(*(realwaste_dataset[i] for i in range(len(realwaste_dataset))))
    all_data = torch.stack(list(all_data))
    all_labels = torch.tensor(list(all_labels))
    return all_data, all_labels, idx_to_class, realwaste_dataset

def tensor_to_np(tensor_data: torch.Tensor) -> np.ndarray:
        return tensor_data.detach().numpy()
    


def preprocess_data(all_data, all_labels, sample_size=None, seed=42):
    """
    Convertit en numpy. Si sample_size est spécifié, effectue un sous-échantillonnage aléatoire.
    Retourne : X_data, y_labels (numpy arrays)
    """
    X_data = tensor_to_np(all_data)
    y_labels = tensor_to_np(all_labels)

    if sample_size is not None:
        np.random.seed(seed)
        num_samples = len(all_data)
        if sample_size > num_samples:
            print(f"Attention : sample_size ({sample_size}) est plus grand que le dataset ({num_samples}). Utilisation de toutes les données.")
            sample_size = num_samples
        indices = np.random.choice(range(num_samples), size=sample_size, replace=False)
        X_data_sample = X_data[indices]
        y_labels_sample = y_labels[indices]
        return X_data_sample, y_labels_sample

    return X_data, y_labels

def split_data_active_learning(X_data, y_labels, n_classes, n_per_class, seed=42, val_size=0.15, val_final_size=0.10):
    """
    Split spécialisé pour l'apprentissage actif avec 4 ensembles :
    - Pool initial équilibré (n_classes * n_per_class)
    - Pool d'acquisition (reste des données d'entraînement)
    - Validation (pour monitoring pendant l'AL)
    - Validation finale (jamais vue pendant l'entraînement, pour évaluation finale)
    
    Args:
        X_data: numpy array des données
        y_labels: numpy array des labels
        n_classes: nombre de classes
        n_per_class: nombre d'échantillons par classe pour le pool initial
        seed: graine aléatoire
        val_size: proportion pour le set de validation (défaut 15%)
        val_final_size: proportion pour le set de validation finale (défaut 10%)
    
    Returns:
        X_initial, y_initial, X_pool, y_pool, X_validation, y_validation, X_val_final, y_val_final
    """
    np.random.seed(seed)
    n_samples = len(X_data)
    indices = np.arange(n_samples)
    
    print(f"Dataset total: {n_samples} échantillons")
    
    # 1. Séparer d'abord la validation finale (jamais vue)
    sss_val_final = StratifiedShuffleSplit(n_splits=1, test_size=val_final_size, random_state=seed)
    rest_idx, val_final_idx = next(sss_val_final.split(indices, y_labels))
    
    X_val_final = X_data[val_final_idx]
    y_val_final = y_labels[val_final_idx]
    X_data_rest = X_data[rest_idx]
    y_labels_rest = y_labels[rest_idx]
    
    print(f"Validation finale: {len(X_val_final)} échantillons ({val_final_size*100:.1f}%)")
    
    # 2. Séparer la validation (monitoring AL)
    val_size_relative = val_size / (1 - val_final_size)  # Ajuster la proportion
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=seed)
    train_idx, val_idx = next(sss_val.split(X_data_rest, y_labels_rest))
    
    X_validation = X_data_rest[val_idx]
    y_validation = y_labels_rest[val_idx]
    X_train_data = X_data_rest[train_idx]
    y_train_labels = y_labels_rest[train_idx]
    
    print(f"Validation (monitoring): {len(X_validation)} échantillons ({val_size*100:.1f}%)")
    print(f"Données d'entraînement restantes: {len(X_train_data)} échantillons")
    
    # 3. Créer le pool initial équilibré à partir des données d'entraînement
    X_initial, y_initial = select_n_per_class(X_train_data, y_train_labels, n_per_class=n_per_class, seed=seed)
    
    # 4. Créer le pool d'acquisition (reste après pool initial)
    # Identifier les indices utilisés pour le pool initial
    initial_indices = []
    for c in np.unique(y_train_labels):
        idx = np.where(y_train_labels == c)[0]
        np.random.seed(seed)  # Même graine que select_n_per_class
        chosen = np.random.choice(idx, size=n_per_class, replace=False)
        initial_indices.extend(chosen)
    
    # Créer un masque pour exclure les indices du pool initial
    mask = np.ones(len(X_train_data), dtype=bool)
    mask[initial_indices] = False
    X_pool = X_train_data[mask]
    y_pool = y_train_labels[mask]
    
    print(f"Pool initial: {len(X_initial)} échantillons ({n_classes} × {n_per_class})")
    print(f"Pool d'acquisition: {len(X_pool)} échantillons")
    
    # Vérification de la cohérence
    total_check = len(X_initial) + len(X_pool) + len(X_validation) + len(X_val_final)
    print(f"Vérification: {total_check} = {n_samples} ✓" if total_check == n_samples else f"❌ Erreur: {total_check} ≠ {n_samples}")
    
    # Vérification de l'équilibrage du pool initial
    print(f"Distribution du pool initial:")
    for c in np.unique(y_initial):
        count = np.sum(y_initial == c)
        print(f"  Classe {c}: {count} échantillons")
    
    return X_initial, y_initial, X_pool, y_pool, X_validation, y_validation, X_val_final, y_val_final


def select_n_per_class(X, y, n_per_class, seed=42):
    """
    Sélectionne n_per_class indices par classe de façon aléatoire.
    X : numpy array (N, ...)
    y : numpy array (N,)
    Retourne : X_sample, y_sample (numpy arrays)
    """
    np.random.seed(seed)
    selected_indices = []
    y = np.array(y)
    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) < n_per_class:
            raise ValueError(f"Pas assez d'images pour la classe {c} (trouvé {len(idx)}, requis {n_per_class})")
        chosen = np.random.choice(idx, size=n_per_class, replace=False)
        selected_indices.extend(chosen)
    selected_indices = np.array(selected_indices)
    return X[selected_indices], y[selected_indices]