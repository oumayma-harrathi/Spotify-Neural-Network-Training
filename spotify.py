# Importation néc..
import matplotlib.pyplot as plt
print("Styles disponibles:", plt.style.available)  # Vérifier les styles valides
plt.style.use('ggplot')  # Remplacement par un style valide
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras import callbacks
except ModuleNotFoundError:
    print("Erreur: TensorFlow n'est pas installé. Veuillez l'installer avant d'exécuter ce script.")
    exit()

spotify = pd.read_csv('../input/dl-course-data/spotify.csv')

X = spotify.copy().dropna()#avoir une copie spotify sans les valeurs manquantes

y = X.pop('track_popularity')#track_popularity est la colonne cible
artists = X['track_artist']#on garde la colonne track_artist pour le group_split

features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']#vals numériques
features_cat = ['playlist_genre']#vals catégorielles

preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)#preparation des features numériques et catégorielles au modèle

#division des données en respectant des groupes (ici les artistes)
#pour éviter de mélanger les artistes dans le train et le test
def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100 # popularity is on a scale 0-100, so this rescales to 0-1.
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))

model = keras.Sequential([
    layers.Dense(1, input_shape=input_shape),
])
#l'optilisateur Adam est un algorithme d'optimisation de gradient stochastique
#qui adapte le taux d'apprentissage pour chaque paramètre
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    verbose=0,
)
#historisation de l'historique d'apprentissage en cas d'arret ou reprendre l'entrainement 
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));

history_df.loc[10:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));

# YOUR CODE HERE: define an early stopping callback
early_stopping = callbacks.EarlyStopping(
    patience=5,  # Nombre d'époques sans amélioration avant l'arrêt
    min_delta=0.001,  # Seuil d'amélioration minimale pour considérer une progression
    restore_best_weights=True
)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),    
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=50,
    callbacks=[early_stopping]
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));