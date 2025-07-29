# main.py
"""
Script principal pour l'apprentissage actif sur la classification d'images de déchets
"""
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import torch
import mlflow
from sklearn.metrics import ConfusionMatrixDisplay
from model_management import save_active_learning_model



# Supprimer les avertissements de dépréciation de sklearn
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import torch.nn as nn # Importer nn ici

from load_data import load_images, preprocess_data, split_data_active_learning
from cnn_model import load_CNN_model
from acquisition_functions import max_entropy, bald, mean_std_acquisition, var_ratios
from active_learning import active_learning


def main():
    data_dir = '../RealWaste'  # Chemin vers le dossier de données originales (non augmentées)
    RESULT_DIR = 'Results'  # Dossier pour sauvegarder les résultats
    

    # Paramètres d'entraînement optimisés
    BATCH_SIZE = 8
    EPOCHS = 100  # Augmenter pour permettre plus d'apprentissage
    MC_DROPOUT_ITER = 20 # Nombre d'itérations pour l'estimation de l'incertitude
    EXPERIMENTS = 1  # Une seule expérience par fonction d'acquisition
    N_QUERY = 10 # Nombre d'images à acquérir à chaque itération
    N_PER_CLASS = 20 # Nombre d'images par classe pour l'échantillonnage équilibré
    # Hyperparamètres principaux documentés pour Airflow/MLflow
    LR = 0.001  # Learning rate fixé à 0.001 comme recommandé
    p = 0.5     # Paramètre p pour la formule du papier
    l = 0.5     # Paramètre l pour la formule du papier (valeur du papier)
    SEED = 350
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    N_CYCLES = 80  # Nombre de cycles d'apprentissage actif pour avoir 1000 de données labellisées


    # Choix du modèle : 'vgg' ou 'resnet'
    MODEL_TYPE = 'vgg'  # <-- Modifiez ici pour 'vgg' ou 'resnet'

    print(f"Configuration d'entraînement:")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Epochs: {EPOCHS}")
    print(f"- Device: {DEVICE}")
    print(f"- Modèle utilisé: {MODEL_TYPE}")
    print(f"- Utilisation de toutes les données disponibles")

    # Configuration MLflow - Créer ou définir l'expérience
    experiment_name = "Active_Learning_Waste_Classification_vgg16_full_dataset_24_07_v2"
    try:
        # Essayer de créer une nouvelle expérience
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Nouvelle expérience créée: {experiment_name} (ID: {experiment_id})")
    except mlflow.exceptions.MlflowException:
        # L'expérience existe déjà, la récupérer
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Utilisation de l'expérience existante: {experiment_name} (ID: {experiment_id})")
    
    # Définir l'expérience active
    mlflow.set_experiment(experiment_name)

    # Créer le dossier de résultats s'il n'existe pas
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)


    # Chargement des données
    print("\n=== Chargement des données ===")
    all_data, all_labels, idx_to_class, realwaste_dataset = load_images(data_dir, augment=False)
    X_data, y_labels = preprocess_data(all_data, all_labels, sample_size=None, seed=SEED)
    print(f"Dataset complet chargé: {len(X_data)} images, {len(realwaste_dataset.classes)} classes")
    

    # Split stratifié avec pool initial équilibré (n_classes * n_per_class) et validation finale
    N_CLASS = len(realwaste_dataset.classes)
    n_classes = N_CLASS  # Correction pour l'usage plus bas
    X_initial, y_initial, X_pool, y_pool, X_validation, y_validation, X_val_final, y_val_final = split_data_active_learning(
        X_data, y_labels, seed=SEED, n_classes=N_CLASS, n_per_class=N_PER_CLASS)

    print(f"Répartition des données:")
    print(f"- Pool initial: {len(X_initial)} images")
    print(f"- Pool d'acquisition: {len(X_pool)} images")
    print(f"- Données de Validation Entrainement: {len(X_validation)} images")
    print(f"- Données de Validation finale: {len(X_val_final)} images")

    # Calcul du weight decay (l2) selon la formule (1-p)*l^2/N
    N = len(X_initial)
    Weight_Decay = (1 - p) * (l ** 2) / N

    # Entraînement avec les 4 fonctions d'acquisition
    ACQ_FUNCS = [max_entropy, bald, mean_std_acquisition, var_ratios]
    results = dict()
    losses = dict()
    train_losses = dict()  # Pour stocker les pertes de validation
    aucs = dict()  # Pour stocker les courbes AUC de chaque fonction d'acquisition
    auc_curves = dict()  # Pour stocker les courbes AUC de chaque fonction d'acquisition
    val_acc_curves = dict()  # Pour stocker les courbes validation accuracy de chaque fonction d'acquisition


    for acq_func in ACQ_FUNCS:
        acq_func_name = acq_func.__name__
        with mlflow.start_run(run_name=f"{acq_func_name}_full_dataset"):
            print(f"\n=== Entraînement avec {acq_func_name} ===")

            # Log des paramètres
            mlflow.log_param("acquisition_function", acq_func_name)
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("n_query", N_QUERY)
            mlflow.log_param("lr", LR)
            mlflow.log_param("weight_decay", Weight_Decay)
            mlflow.log_param("p", p)
            mlflow.log_param("l", l)
            mlflow.log_param("n_cycles", N_CYCLES)
            mlflow.log_param("n_per_class", N_PER_CLASS)
            mlflow.log_param("mc_dropout_iter", MC_DROPOUT_ITER)
            mlflow.log_param("experiments", EXPERIMENTS)
            mlflow.log_param("seed", SEED)
            mlflow.log_param("dataset_size", len(X_data))
            mlflow.log_param("n_classes", n_classes)
            mlflow.log_param("initial_pool_size", len(X_initial))
            mlflow.log_param("validation_size", len(X_validation))
            mlflow.log_param("test_size", len(X_val_final))
            mlflow.log_param("device", str(DEVICE))

            # Tags pour faciliter la recherche
            mlflow.set_tag("model_type", "ResNet18" if MODEL_TYPE == 'resnet' else "VGG16")
            mlflow.set_tag("task", "active_learning")
            mlflow.set_tag("domain", "waste_classification")
            mlflow.set_tag("dataset", "full")
            mlflow.set_tag("date", "2024-07-18")  # Date de l'entraînement

            # Copier les pools pour cette fonction d'acquisition
            X_pool_copy = X_pool.copy()
            y_pool_copy = y_pool.copy()


     
            estimator = load_CNN_model(n_classes, DEVICE, LR, BATCH_SIZE, EPOCHS, Weight_Decay, class_weight=None)

            # Utiliser la nouvelle signature de active_learning
            training_hist, val_losses, model_accuracy_test, f1, precision, recall, learner, classif_report, cm, auc_scores, auc_curve, train_losses_curve, val_accuracies_curve = active_learning(
                query_strategy=acq_func,
                x_validation=X_validation,
                y_validation=y_validation,
                X_test=X_val_final,
                y_test=y_val_final,
                X_pool=X_pool_copy,
                y_pool=y_pool_copy,
                X_initial=X_initial,
                y_initial=y_initial,
                model=estimator,
                n_iterations=N_CYCLES, # Nombre de cycles d'apprentissage actif
                n_query=N_QUERY,
                mc_dropout_iterations=MC_DROPOUT_ITER, # Pour l'estimation de l'incertitude
                device=DEVICE, 
                target_names=list(idx_to_class.values())  # Passer les noms de classes pour le rapport de classification
            )

            # Log uniquement les métriques calculées dans la boucle d'entraînement
            mlflow.log_metric("active_learning_final_accuracy", model_accuracy_test)
            mlflow.log_metric("active_learning_final_f1", f1)
            mlflow.log_metric("active_learning_final_precision", precision)
            mlflow.log_metric("active_learning_final_recall", recall)

            # Log des auc_scores dans MLflow
            for auc_type, auc_value in auc_scores.items():
                mlflow.log_metric(f"auc_{auc_type}", auc_value)


            # Log de la courbe d'AUC, val_losses et train_losses à chaque itération
            for i, (auc_val, val_loss, train_loss_value) in enumerate(zip(auc_curve, val_losses, train_losses_curve)):
                mlflow.log_metric("val_auc", float(auc_val), step=i)
                mlflow.log_metric("val_loss", float(val_loss), step=i)
                mlflow.log_metric("train_loss", float(train_loss_value), step=i)
                # Log de l'accuracy validation (training_hist) et test (val_accuracies_curve) à chaque itération
                if i < len(training_hist):
                    mlflow.log_metric("test_accuracy", float(training_hist[i]), step=i)
                if val_accuracies_curve is not None and i < len(val_accuracies_curve):
                    mlflow.log_metric("val_accuracy", float(val_accuracies_curve[i]), step=i)

            # Log du rapport de classification comme artefact texte
            classif_report_path = os.path.join(RESULT_DIR, f"classification_report_{acq_func_name}.txt")
            with open(classif_report_path, "w", encoding="utf-8") as f:
                f.write(classif_report)
            mlflow.log_artifact(classif_report_path)

            # Log de la matrice de confusion comme image et CSV

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(idx_to_class.values()))
            plt.figure(figsize=(10, 8))
            disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
            plt.title(f"Matrice de confusion - {acq_func_name}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            cm_img_path = os.path.join(RESULT_DIR, f"confusion_matrix_{acq_func_name}.png")
            plt.savefig(cm_img_path)
            plt.close()
            mlflow.log_artifact(cm_img_path)

            # Logguer la matrice de confusion sous forme de tableau (CSV)
            cm_df = pd.DataFrame(cm, index=list(idx_to_class.values()), columns=list(idx_to_class.values()))
            cm_csv_path = os.path.join(RESULT_DIR, f"confusion_matrix_{acq_func_name}.csv")
            cm_df.to_csv(cm_csv_path)
            mlflow.log_artifact(cm_csv_path)

            # Stocker la dernière valeur de la courbe d'AUC pour la courbe comparative
            aucs[acq_func_name] = auc_curve[-1] if len(auc_curve) > 0 else 0.0


            # Sauvegarder le modèle complet avec la fonction dédiée
            metrics_dict = {
                'final_accuracy': model_accuracy_test,
                'final_f1': f1,
                'final_precision': precision,
                'final_recall': recall,
                'auc_curve': auc_curve,
                'auc_scores': auc_scores,
                'training_history': training_hist,
                'val_accuracies_curve': val_accuracies_curve
            }
            save_active_learning_model(learner, metrics_dict, model_name=f"learner_{acq_func_name}_full")

            results[acq_func_name] = training_hist
            losses[acq_func_name] = val_losses
            train_losses[acq_func_name] = train_losses_curve
            val_acc_curves[acq_func_name] = val_accuracies_curve


    print("\n=== Entraînement terminé pour toutes les fonctions d'acquisition ===")


    # --- Graphique comparatif des validation accuracy ---
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    for acq_name, val_acc_curve in val_acc_curves.items():
        smoothed_val_acc = gaussian_filter1d(val_acc_curve, sigma=0.9)
        plt.plot(smoothed_val_acc, label=acq_name, linewidth=2)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Courbes de Validation Accuracy - Comparaison des Fonctions d\'Acquisition', fontsize=14)
    plt.legend(title="Fonctions d'Acquisition", fontsize=10, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    test_acc_comparison_filename = "comparison_all_acquisition_functions_validation_accuracy.png"
    plt.savefig(os.path.join(RESULT_DIR, test_acc_comparison_filename))
    plt.show()

    # --- Graphique comparatif des training history (accuracy) ---
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    for acq_name, training_hist in results.items():
        smoothed_acc = gaussian_filter1d(training_hist, sigma=0.9)
        plt.plot(smoothed_acc, label=acq_name, linewidth=2)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Courbes de Training History (Accuracy) - Comparaison des Fonctions d\'Acquisition', fontsize=14)
    plt.legend(title="Fonctions d'Acquisition", fontsize=10, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    training_hist_comparison_filename = "comparison_all_acquisition_functions_training_history.png"
    plt.savefig(os.path.join(RESULT_DIR, training_hist_comparison_filename))
    plt.show()

    # --- Graphique comparatif des courbes AUC ---
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    for acq_name, auc_curve in auc_curves.items():
        smoothed_auc = gaussian_filter1d(auc_curve, sigma=0.9)
        plt.plot(smoothed_auc, label=acq_name, linewidth=2)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title('Courbes AUC - Comparaison des Fonctions d\'Acquisition', fontsize=14)
    plt.legend(title="Fonctions d'Acquisition", fontsize=10, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    auc_curve_comparison_filename = "comparison_all_acquisition_functions_auc_curves.png"
    plt.savefig(os.path.join(RESULT_DIR, auc_curve_comparison_filename))
    plt.show()

    # --- Graphique comparatif des losses ---
    plt.figure(figsize=(12, 8))
    for key, value in losses.items():
        smoothed_loss = gaussian_filter1d(value, sigma=0.9)
        plt.plot(smoothed_loss, label=key, linewidth=2)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Courbe de Loss Validation - Dataset Complet', fontsize=14)
    plt.legend(title="Fonctions d'Acquisition", fontsize=10, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    loss_plot_filename = "comparison_all_acquisition_functions_loss.png"
    plt.savefig(os.path.join(RESULT_DIR, loss_plot_filename))
    plt.show()

    # --- Tableau récapitulatif final ---
    accuracy_thresholds = [0.6, 0.7, 0.8]
    summary_val = {thresh: {} for thresh in accuracy_thresholds}
    summary_test = {thresh: {} for thresh in accuracy_thresholds}
    initial_size = len(X_initial)

    # Test accuracy (training_hist)
    for acq_name, acc_curve in results.items():
        acc_curve = np.array(acc_curve)
        for thresh in accuracy_thresholds:
            passed_indices = np.where(acc_curve >= thresh)[0]
            if len(passed_indices) > 0:
                idx = passed_indices[0]
                n_acquired = initial_size + idx * N_QUERY
                summary_test[thresh][acq_name] = n_acquired
            else:
                summary_test[thresh][acq_name] = 'Non atteint'

    # Validation accuracy (val_acc_curves)
    for acq_name, acc_curve in val_acc_curves.items():
        acc_curve = np.array(acc_curve)
        for thresh in accuracy_thresholds:
            passed_indices = np.where(acc_curve >= thresh)[0]
            if len(passed_indices) > 0:
                idx = passed_indices[0]
                n_acquired = initial_size + idx * N_QUERY
                summary_val[thresh][acq_name] = n_acquired
            else:
                summary_val[thresh][acq_name] = 'Non atteint'

    summary_test_df = pd.DataFrame(summary_test).T
    summary_test_df.index.name = 'Accuracy Threshold (Test)'

    summary_val_df = pd.DataFrame(summary_val).T
    summary_val_df.index.name = 'Accuracy Threshold (Validation)'

    print(f"\n=== Tableau récapitulatif - Validation Accuracy ===")
    print(summary_val_df)
    print(f"\n=== Tableau récapitulatif - Test Accuracy ===")
    print(summary_test_df)

    # Sauvegarder les tableaux finaux
    summary_val_filename = "summary_comparison_full_dataset_validation.csv"
    summary_test_filename = "summary_comparison_full_dataset_test.csv"
    summary_val_df.to_csv(os.path.join(RESULT_DIR, summary_val_filename))
    summary_test_df.to_csv(os.path.join(RESULT_DIR, summary_test_filename))

    print("\n" + "="*60)
    print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    print(f"📁 Résultats sauvegardés dans: {RESULT_DIR}/")
    print("📊 4 modèles entraînés (un par fonction d'acquisition)")
    print("📈 Graphique et tableau de comparaison générés")
    print("🔬 Suivez les résultats sur MLflow UI: http://localhost:5000")
    print("="*60)

if __name__ == "__main__":
    main()

