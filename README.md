# Apprentissage Actif pour la Classification des Déchets

## Problème énoncé dans l'article

L'apprentissage actif est une approche semi-supervisée qui vise à réduire la quantité de données annotées nécessaires pour entraîner un modèle performant. Plutôt que de s'appuyer sur un grand jeu d'entraînement exhaustivement labellisé, il fait intervenir un annotateur (« oracle ») tout au long du processus :

* Une fonction d'acquisition, généralement fondée sur l'incertitude du modèle, identifie les exemples les plus informatifs dans un pool de données non étiquetées
* Ces exemples sont alors transmis à l'oracle
* Leurs annotations viennent enrichir le jeu d'entraînement
* Le modèle est réentraîné sur cet ensemble actualisé

Ce paradigme se révèle particulièrement prometteur en apprentissage profond, où la montée en puissance des modèles s'est historiquement faite au prix de jeux de données massifs et coûteux à annoter. Pourtant, deux obstacles freinent aujourd'hui son adoption :

1. La difficulté à gérer des données à très haute dimensionnalité (images, séquences, etc.)
2. L'absence native d'une mesure d'incertitude fiable dans les réseaux de neurones profonds

Des travaux récents ont montré comment l'incertitude du modèle peut être prise en compte dans l'apprentissage profond en adoptant une approche bayésienne avec dropout dans les réseaux neuronaux. L'article "Deep Bayesian Active Learning with Image Data" adresse précisément ces défis : il propose d'incorporer un cadre bayésien (via le dropout stochastique) au sein de réseaux convolutifs pour quantifier l'incertitude et piloter des stratégies d'acquisition actives sur des problèmes de classification d'images.

## Problème choisi et données associées

Mon objectif est d'exploiter le jeu de données RealWaste (UCI ML Repository) et le framework d'apprentissage actif présenté par Gal et al. pour évaluer dans quelle mesure ce paradigme peut améliorer la classification automatique des déchets.

### Contexte et motivation

Dans un contexte de réchauffement climatique, le tri des déchets joue un rôle crucial :

* Il permet de valoriser les matériaux
* Il réduit la quantité de déchets envoyés en décharge
* Il atténue l'empreinte environnementale de nos modes de consommation

Or, entraîner un modèle capable de reconnaître tous les types de déchets exigerait un vaste corpus annoté, dont la constitution est à la fois longue et coûteuse.

### Approche proposée

L'apprentissage actif offre ici une solution prometteuse : en sollicitant intelligemment l'annotateur (oracle) pour ne qualifier que les exemples les plus informatifs — ceux dont l'incertitude prédictive est la plus élevée — on limite drastiquement le volume d'annotations nécessaires tout en conservant un haut niveau de performance.

Ce travail vise donc à quantifier les gains (en termes de nombre d'étiquettes et de précision) obtenus en appliquant cette stratégie au problème de classification des déchets sur RealWaste.