# active_learning.py
"""
Boucle d'apprentissage actif utilisant modAL
"""
from modAL.models import ActiveLearner
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import torch
import torch.nn as nn

def get_class_weights(y_train, device='cpu'):
    """
    Calcule les poids des classes pour données déséquilibrées
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    import torch
    
    classes = np.unique(y_train)
    weights = compute_class_weight(
        'balanced', 
        classes=classes, 
        y=y_train
    )
    
    # Conversion en tensor PyTorch sur le bon device
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    return class_weights

def calculate_auc_scores(y_true, y_pred_proba, n_classes):
    """
    Calcule l'AUC pour classification binaire et multiclass
    """
    auc_scores = {}
    
    if n_classes == 2:
        # Classification binaire
        auc_scores['binary_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        # Classification multiclass - One-vs-Rest
        try:
            auc_scores['macro_auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                   multi_class='ovr', average='macro')
            auc_scores['weighted_auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                      multi_class='ovr', average='weighted')
        except ValueError as e:
            print(f"Impossible de calculer l'AUC multiclass: {e}")
            auc_scores['macro_auc'] = 0.0
            auc_scores['weighted_auc'] = 0.0
    
    return auc_scores

def active_learning(query_strategy, model, X_test, y_test, 
                    x_validation, y_validation, X_pool, y_pool, X_initial, y_initial, 
                    n_query=10, n_iterations=15, mc_dropout_iterations=15, 
                    device=None, target_names=None):
    
    learner = ActiveLearner(
        estimator=model,
        query_strategy=query_strategy,
        X_training=X_initial,
        y_training=y_initial,
        
    )
    
    # Score initial avant la boucle d'apprentissage actif
    learner.estimator.module_.eval() # S'assurer que le modèle est en mode évaluation
    accuracy_scores = [learner.score(X_test, y_test)]
    val_accuracies = [learner.score(X_test, y_test)]
    test_losses = [0]
    train_losses = [0]
    auc_curve = [0]
    
    print(f"Démarrage de l'apprentissage actif avec {n_iterations} itérations")
    print(f"Accuracy initiale (test): {accuracy_scores[0]:.4f}")
    
    # Boucle d'apprentissage actif
    for i in range(n_iterations):
        print(f"\n--- Itération {i+1}/{n_iterations} ---")
        print(f"Taille du pool: {len(X_pool)}")

        if len(X_pool) < n_query:
            print(f"Pool d'acquisition épuisé à l'itération {i}. Arrêt de l'apprentissage actif.")
            break
        
        current_y_train = learner.y_training
        class_weights = get_class_weights(current_y_train, device=device)

        # Afficher la distribution courante (amélioré)
        unique, counts = np.unique(current_y_train, return_counts=True)
        print("Distribution courante:")
        for u, c in zip(unique, counts):
            print(f"  Classe {int(u)}: {int(c)} exemples")
        print(f"Class weights: {[float(w) for w in class_weights.cpu().numpy()]}")

        # 2. Créer le critère avec les nouveaux poids
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        learner.estimator.set_params(criterion__weight=class_weights)
            
        # Passer le nombre d'itérations de dropout à la stratégie de query
        query_idx, _ = learner.query(
            X_pool, n_instances=n_query, 
            n_iterations=mc_dropout_iterations,
            training=True
        )   
        
        # Le modèle est automatiquement mis en mode entraînement par .teach()
        learner.teach(X_pool[query_idx], y_pool[query_idx])
        
        # Mettre à jour le pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
        
        # Mettre le modèle en mode évaluation pour la validation
        learner.estimator.module_.eval()
        
        model_accuracy_test = learner.score(X_test, y_test)
        accuracy_scores.append(model_accuracy_test)
        print(f"Accuracy test: {model_accuracy_test:.4f}")

        # Calculer et stocker l'accuracy sur le jeu de test à chaque itération
        model_accuracy_validation_iter = learner.score(x_validation, y_validation)
        val_accuracies.append(model_accuracy_validation_iter)
        print(f"Accuracy validation: {model_accuracy_validation_iter:.4f}")

        # Calcul de la loss de validation et de la training loss

        with torch.no_grad():
            # Test loss
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(learner.estimator.device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(learner.estimator.device)
            outputs_test = learner.estimator.forward(X_test_tensor, training=False)
            outputs_test = outputs_test.to(learner.estimator.device)
            test_loss_value = criterion(outputs_test, y_test_tensor).item()
            test_losses.append(test_loss_value)
            print(f"Loss test: {test_loss_value:.4f}")

            # Training loss
            X_train_tensor = torch.tensor(learner.X_training, dtype=torch.float32).to(learner.estimator.device)
            y_train_tensor = torch.tensor(learner.y_training, dtype=torch.long).to(learner.estimator.device)
            outputs_train = learner.estimator.forward(X_train_tensor, training=False)
            outputs_train = outputs_train.to(learner.estimator.device)
            train_loss_value = criterion(outputs_train, y_train_tensor).item()
            train_losses.append(train_loss_value)
            print(f"Loss training: {train_loss_value:.4f}")

            # Calcul de l'AUC à chaque itération sur le set de validation
            X_val_tensor = torch.tensor(x_validation, dtype=torch.float32).to(learner.estimator.device)
            outputs_val = learner.estimator.forward(X_val_tensor, training=False)
            y_val_pred_proba = outputs_val.softmax(dim=1).cpu().numpy()
            n_val_classes = len(np.unique(y_validation))
            y_val_auc = y_validation
            if n_val_classes > 2:
                y_val_auc = label_binarize(y_validation, classes=np.arange(n_val_classes))
            try:
                if n_val_classes == 2:
                    auc_val = roc_auc_score(y_val_auc, y_val_pred_proba[:, 1])
                else:
                    auc_val = roc_auc_score(y_val_auc, y_val_pred_proba, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"AUC non calculable à l'itération {i+1}: {e}")
                auc_val = 0.0
            auc_curve.append(auc_val)
            print(f"AUC validation: {auc_val:.4f}")
            
    print(f"\nFin de l'apprentissage actif après {len(accuracy_scores)-1} itérations")

    print("--- Evaluration du modele d'apprentissage actif ---")
    
    # Mettre le modèle en mode évaluation pour le test final
    learner.estimator.module_.eval()
    model_accuracy_test = learner.score(X_test, y_test)
    print(f"Accuracy finale (training): {model_accuracy_test:.4f}")

    # Calcul des probabilités pour l'AUC
    y_pred_proba = learner.predict_proba(X_test)
    n_classes = len(np.unique(y_test))

    # Pour le multiclass, s'assurer que y_test est binarisé si besoin
    y_test_auc = y_test
    if n_classes > 2:
        # y_test doit être binarisé pour roc_auc_score multiclass
        y_test_auc = label_binarize(y_test, classes=np.arange(n_classes))

    # Calcul de l'AUC
    auc_scores = calculate_auc_scores(y_test_auc, y_pred_proba, n_classes)

    # Calcul d'autres métriques de classification
    y_pred = learner.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    classif_report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Affichage des scores AUC
    for auc_type, auc_value in auc_scores.items():
        print(f"{auc_type}: {auc_value:.4f}")

    print(f"F1-score (weighted): {f1:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")

    # Retourne aussi la courbe d'accuracy sur le test
    return accuracy_scores, test_losses, model_accuracy_test, f1, precision, recall, learner, classif_report, cm, auc_scores, auc_curve, train_losses, val_accuracies
