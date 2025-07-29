# Les fonctions et classes ont été déplacées dans les modules :
# - acquisition_functions.py
# - cnn_model.py
# - load_data.py
# - active_learning.py
# Le script principal est désormais dans main.py

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from skorch import NeuralNetClassifier
from modAL.models import ActiveLearner
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

# Supprimer les avertissements de dépréciation de sklearn
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# --- Conversion d'un tenseur torch en numpy array ---
def tensor_to_np(tensor_data: torch.Tensor) -> np.ndarray:
    return tensor_data.detach().numpy()

# --- Chargement et préparation des données ---
def load_data(data_dir, sample_size=1000, batch_size=32, seed=42):
    # Définir les transformations pour les images
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Charger le dataset d'images
    realwaste_dataset = ImageFolder(root=data_dir, transform=data_transforms)
    # Créer un mapping index -> nom de classe
    idx_to_class = {v: k for k, v in realwaste_dataset.class_to_idx.items()}
    # Extraire toutes les images et labels
    all_data, all_labels = zip(*(realwaste_dataset[i] for i in range(len(realwaste_dataset))))
    all_data = torch.stack(list(all_data))
    all_labels = torch.tensor(list(all_labels))
    # Conversion en numpy arrays
    X_data = tensor_to_np(all_data)
    y_labels = tensor_to_np(all_labels)
    # Sélectionner un sous-échantillon aléatoire pour le prototypage
    np.random.seed(seed)
    indices = np.random.choice(range(len(all_data)), size=sample_size, replace=False)
    X_data_sample = X_data[indices]
    y_labels_sample = y_labels[indices]
    # Split stratifié pour respecter la distribution des classes
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed, train_size=sample_size)
    for train_index, _ in sss.split(X_data, y_labels):
        X_data_sample_stratified = X_data[train_index]
        y_labels_sample_stratified = y_labels[train_index]
    # Split en train/test/validation
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_data_sample_stratified, y_labels_sample_stratified, test_size=0.2, random_state=50, stratify=y_labels_sample)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=seed, stratify=y_temp)
    # Sélection d'un ensemble initial pour l'apprentissage actif
    initial_idx = np.random.choice(range(len(X_train)), size=100, replace=False)
    X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
    X_pool, y_pool = np.delete(X_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)
    return (X_initial, y_initial, X_pool, y_pool, X_validation, y_validation, X_test, y_test, idx_to_class, realwaste_dataset)

# --- Fonctions d'acquisition pour l'apprentissage actif ---
def predictions_from_pool(model, X_pool, T=100, training=True):
    # Prédictions MC Dropout sur un sous-ensemble aléatoire du pool
    subset_size = min(500, len(X_pool))
    random_subset = np.random.choice(range(len(X_pool)), size=subset_size, replace=False)
    with torch.no_grad():
        outputs = np.stack([
            torch.softmax(model.estimator.forward(X_pool[random_subset], training=training), dim=-1).cpu().numpy()
            for _ in range(T)
        ])
    return outputs, random_subset

def shannon_entropy_function(model, X_pool, T=100, E_H=False, training=True):
    # Calcul de l'entropie de Shannon (incertitude prédictive)
    outputs, random_subset = predictions_from_pool(model, X_pool, T, training=training)
    pc = outputs.mean(axis=0)
    H = (-pc * np.log(pc + 1e-10)).sum(axis=-1)
    if E_H:
        E = -np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)
        return H, E, random_subset
    return H, random_subset

def max_entropy(model, X_pool, n_query=10, T=100, training=True):
    # Sélectionne les points du pool avec l'entropie prédictive maximale
    acquisition, random_subset = shannon_entropy_function(model, X_pool, T, training=training)
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]

def mean_std_acquisition(model, X_pool, n_query=10, T=100, training=True):
    # Sélectionne les points avec l'écart-type moyen maximal des prédictions
    outputs, random_subset = predictions_from_pool(model, X_pool, T, training=training)
    expected_p_c = np.mean(outputs, axis=0)
    expected_p_c_squared = np.mean(outputs**2, axis=0)
    sigma_c = expected_p_c_squared - (expected_p_c**2)
    acquisition_scores = np.mean(sigma_c, axis=-1)
    idx = (-acquisition_scores).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]

def bald(model, X_pool, n_query=10, T=100, training=True):
    # Sélectionne les points qui maximisent l'information mutuelle (BALD)
    H, E_H, random_subset = shannon_entropy_function(model, X_pool, T, E_H=True, training=training)
    acquisition = H - E_H
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]

def var_ratios(model, X_pool, n_query=10, T=100, training=True):
    # Sélectionne les points avec le plus faible taux de confiance (variational ratios)
    outputs, random_subset = predictions_from_pool(model, X_pool, T, training)
    preds = np.argmax(outputs, axis=2)
    _, count = stats.mode(preds, axis=0)
    acquisition = (1 - count / preds.shape[1]).reshape((-1,))
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]

# --- Définition et chargement du modèle CNN (VGG16) ---
def load_CNN_model(n_classes, device, lr, batch_size, epochs, weight_decay, pretrained=True):
    if pretrained:
        # Charger VGG16 pré-entraîné et adapter la dernière couche
        vgg16_pretrained = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for param in vgg16_pretrained.parameters():
            param.requires_grad = False
        num_features = vgg16_pretrained.classifier[6].in_features
        vgg16_pretrained.classifier[6] = nn.Linear(num_features, n_classes)
        for param in vgg16_pretrained.classifier[6].parameters():
            param.requires_grad = True
        model = vgg16_pretrained.to(device)
    else:
        class VGG16(nn.Module):
            def __init__(self, n_classes):
                super().__init__()
                # ... (architecture as in notebook) ...
        model = VGG16(n_classes).to(device)
    # Wrapper skorch pour l'entraînement sklearn-like
    cnn_classifier = NeuralNetClassifier(
        module=model,
        lr=lr,
        batch_size=batch_size,
        max_epochs=epochs,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
        device=device,
        optimizer__weight_decay=weight_decay,
    )
    return cnn_classifier

# --- Boucle d'apprentissage actif ---
def active_learning(query_strategy, model, x_validation, y_validation, X_test, y_test, X_pool, y_pool, X_initial, y_initial, n_query=10, n_iterations=15, training=True):
    # Initialisation de l'ActiveLearner (modAL)
    learner = ActiveLearner(
        estimator=model,
        query_strategy=query_strategy,
        X_training=X_initial,
        y_training=y_initial,
    )
    accuracy_scores = [learner.score(X_test, y_test)]
    for i in range(n_iterations):
        # Sélectionne les points à annoter selon la stratégie d'acquisition
        query_idx, _ = learner.query(X_pool, n_query=n_query, T=n_iterations, training=training)
        learner.teach(X_pool[query_idx], y_pool[query_idx])
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
        model_accuracy_val = learner.score(x_validation, y_validation)
        accuracy_scores.append(model_accuracy_val)
    model_accuracy_test = learner.score(X_test, y_test)
    return accuracy_scores, model_accuracy_test

# --- Script principal d'entraînement ---
def main():
    data_dir = '../RealWaste'  # Chemin vers le dossier de données
    RESULT_DIR = 'result_npy'  # Dossier pour sauvegarder les résultats
    BATCH_SIZE = 32
    DROPOUT_ITER = 15
    EPOCHS = 20
    EXPERIMENTS = 2
    N_QUERY = 10
    LR = 1e-3
    SEED = 350
    p = 0.5
    l2 = 1e-4
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Chargement et préparation des données
    X_initial, y_initial, X_pool, y_pool, X_validation, y_validation, X_test, y_test, idx_to_class, realwaste_dataset = load_data(data_dir, sample_size=1000, batch_size=BATCH_SIZE, seed=SEED)
    n_classes = len(realwaste_dataset.classes)
    N = len(X_initial)
    Weight_Decay = (1-p)*l2/N
    ACQ_FUNCS = [max_entropy, bald, mean_std_acquisition, var_ratios]
    results = dict()
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)
    # Boucle sur les fonctions d'acquisition
    for acq_func in ACQ_FUNCS:
        avg_hist = []
        test_scores = []
        acq_func_name = acq_func.__name__
        print(f"\n---------- Start {acq_func_name} training! ----------")
        for e in range(EXPERIMENTS):
            estimator = load_CNN_model(n_classes, DEVICE, LR, BATCH_SIZE, EPOCHS, Weight_Decay, pretrained=True)
            training_hist, test_score = active_learning(
                query_strategy=acq_func,
                x_validation=X_validation,
                y_validation=y_validation,
                X_test=X_test,
                y_test=y_test,
                X_pool=X_pool,
                y_pool=y_pool,
                X_initial=X_initial,
                y_initial=y_initial,
                model=estimator,
                n_iterations=DROPOUT_ITER,
                n_query=N_QUERY
            )
            avg_hist.append(training_hist)
            test_scores.append(test_score)
        avg_hist = np.average(np.array(avg_hist), axis=0)
        avg_test = sum(test_scores) / len(test_scores)
        print(f"Average Test score for {acq_func_name}: {avg_test}")
        results[acq_func_name] = avg_hist
        np.save(os.path.join(RESULT_DIR, acq_func_name+".npy"), avg_hist)
    print("--------------- Done Training! ---------------")
    # Affichage des courbes de validation
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    for key in results.keys():
        smoothed_data = gaussian_filter1d(results[key], sigma=0.9)
        raw_data = np.insert(smoothed_data, 0, 0.0)
        plt.plot(raw_data, label=key, linewidth=2)
    plt.ylim([0.0, 1.00])
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Active Learning Performance by Acquisition Function', fontsize=14)
    plt.legend(title="Acquisition Functions", fontsize=10, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # --- Génération du tableau récapitulatif : nombre d'images acquises pour atteindre un certain niveau d'accuracy ---
    # Définir les seuils d'accuracy à tester
    accuracy_thresholds = [0.6, 0.7, 0.8, 0.85, 0.9]

    # Dictionnaire des fichiers de résultats pour chaque fonction d'acquisition
    acq_func_files = {
        'max_entropy': 'max_entropy.npy',
        'bald': 'bald.npy',
        'mean_std_acquisition': 'mean_std_acquisition.npy',
        'var_ratios': 'var_ratios.npy',
    }

    results_dir = RESULT_DIR
    acquired_per_iter = N_QUERY  # Nombre d'images acquises à chaque itération
    initial_size = len(X_initial)

    summary = {thresh: {} for thresh in accuracy_thresholds}

    for acq_name, filename in acq_func_files.items():
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            print(f"Fichier manquant pour {acq_name}: {filename}")
            continue
        acc_curve = np.load(path)
        for thresh in accuracy_thresholds:
            # Trouver la première itération où l'accuracy dépasse le seuil
            idx = np.argmax(acc_curve >= thresh)
            if acc_curve[idx] < thresh:
                summary[thresh][acq_name] = 'Non atteint'
            else:
                # Nombre total d'images acquises = initial + (itération * batch)
                n_acquired = initial_size + idx * acquired_per_iter
                summary[thresh][acq_name] = n_acquired

    # Générer le DataFrame pour affichage
    summary_df = pd.DataFrame(summary).T
    summary_df.index.name = 'Accuracy Threshold'
    print("\nTableau récapitulatif : Nombre d'images acquises pour atteindre chaque seuil d'accuracy")
    print(summary_df)

    # (Optionnel) Exporter en CSV
    # summary_df.to_csv('summary_table.csv')

if __name__ == "__main__":
    main()
