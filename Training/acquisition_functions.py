# acquisition_functions.py
"""
Fonctions d'acquisition pour l'apprentissage actif (modAL)
"""
import numpy as np
from scipy import stats
import torch

def predictions_from_pool(model, X_pool, n_iterations=100, training=True):
    # Prédictions MC Dropout sur un sous-ensemble aléatoire du pool
    subset_size = min(500, len(X_pool))
    random_subset = np.random.choice(range(len(X_pool)), size=subset_size, replace=False)
    with torch.no_grad():
        outputs = np.stack([
            torch.softmax(model.estimator.forward(X_pool[random_subset], training=training), dim=-1).cpu().numpy()
            for _ in range(n_iterations)
        ])
    return outputs, random_subset

def shannon_entropy_function(model, X_pool, n_iterations=100, E_H=False, training=True):
    # Calcul de l'entropie de Shannon (incertitude prédictive)
    outputs, random_subset = predictions_from_pool(model, X_pool, n_iterations, training=training)
    pc = outputs.mean(axis=0)
    H = (-pc * np.log(pc + 1e-10)).sum(axis=-1)
    if E_H:
        E = -np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)
        return H, E, random_subset
    return H, random_subset

def max_entropy(model, X_pool, n_instances=10, n_iterations=100, training=True, **kwargs):
    # Sélectionne les points du pool avec l'entropie prédictive maximale
    acquisition, random_subset = shannon_entropy_function(model, X_pool, n_iterations, training=training)
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]

def mean_std_acquisition(model, X_pool, n_instances=10, n_iterations=100, training=True, **kwargs):
    # Sélectionne les points avec l'écart-type moyen maximal des prédictions
    outputs, random_subset = predictions_from_pool(model, X_pool, n_iterations, training=training)
    expected_p_c = np.mean(outputs, axis=0)
    expected_p_c_squared = np.mean(outputs**2, axis=0)
    sigma_c = expected_p_c_squared - (expected_p_c**2)
    acquisition_scores = np.mean(sigma_c, axis=-1)
    idx = (-acquisition_scores).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]

def bald(model, X_pool, n_instances=10, n_iterations=100, training=True, **kwargs):
    # Sélectionne les points qui maximisent l'information mutuelle (BALD)
    H, E_H, random_subset = shannon_entropy_function(model, X_pool, n_iterations, E_H=True, training=training)
    acquisition = H - E_H
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]

def var_ratios(model, X_pool, n_instances=10, n_iterations=100, training=True, **kwargs):
    # Sélectionne les points avec le plus faible taux de confiance (variational ratios)
    outputs, random_subset = predictions_from_pool(model, X_pool, n_iterations, training)
    preds = np.argmax(outputs, axis=2)
    _, count = stats.mode(preds, axis=0)
    acquisition = (1 - count / preds.shape[1]).reshape((-1,))
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]
