# cnn_model.py
"""
Définition et chargement du modèle CNN (VGG16) et ResNet (ResNet18)
"""
import torch
import torch.nn as nn
from torchvision import models
from skorch import NeuralNetClassifier

def load_CNN_model(n_classes, device, lr, batch_size, epochs, weight_decay, class_weight):
    # Charger le VGG16 pré-entraîné
    vgg16_pretrained = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    
    # Geler toutes les couches (convolutives et fully connected)
    for param in vgg16_pretrained.parameters():
        param.requires_grad = False

    # Remplacer seulement la dernière couche de classification
    num_features = vgg16_pretrained.classifier[6].in_features
    vgg16_pretrained.classifier[6] = nn.Linear(num_features, n_classes)

    # Débloquer seulement la dernière couche
    for param in vgg16_pretrained.classifier[6].parameters():
        param.requires_grad = True

    model = vgg16_pretrained.to(device)

    # Créer le classificateur skorch
    cnn_classifier = NeuralNetClassifier(
        module=model,
        lr=lr,
        batch_size=batch_size,
        max_epochs=epochs,
        criterion=nn.CrossEntropyLoss,
        criterion__weight=class_weight,  # Utiliser les poids de classe équilibrés
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
        device=device,
        optimizer__weight_decay=weight_decay,
    )
    return cnn_classifier



