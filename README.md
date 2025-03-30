Ce projet entraîne un modèle de réseau de neurones pour prédire la popularité des morceaux de musique sur Spotify en utilisant des caractéristiques audio et de genre. Il utilise TensorFlow et scikit-learn pour le prétraitement des données et la modélisation.

Prérequis

Avant d'exécuter ce script, assurez-vous d'avoir installé les bibliothèques nécessaires.

Installation des dépendances

Utilisez la commande suivante pour installer les packages requis :
pip install pandas numpy matplotlib scikit-learn tensorflow
Données

Le script charge les données depuis un fichier CSV situé à :
../input/dl-course-data/spotify.csv

Fonctionnalités principales

Prétraitement des données :

Normalisation des caractéristiques numériques avec StandardScaler

Encodage des variables catégorielles avec OneHotEncoder

Séparation des données :

Les données sont divisées en ensembles d'entraînement et de validation en respectant les groupes d'artistes pour éviter une contamination des ensembles

Modélisation avec TensorFlow :

Réseau de neurones simple avec des couches denses

Utilisation de l'optimiseur Adam et de la fonction de perte MAE

Ajout du callback EarlyStopping pour arrêter l'entraînement en cas de stagnation

Exécution

Lancez le script en exécutant :
python script.py

Résultats

Le script affiche la courbe de perte (loss) et enregistre l'historique de l'entraînement pour analyser la performance du modèle.
