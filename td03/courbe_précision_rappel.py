import numpy as np
from skimage import io, util
# Chargement des images d'apprentissage
X_train = np.zeros((15000, 576))
for i in range(3000):
    I = io.imread('imageface/train/pos/%05d.png'%(i+1))
    X_train[i,:] = util.img_as_float(I).flatten()
for i in range(12000):
    I = io.imread('imageface/train/neg/%05d.png'%(i+1))
    X_train[i+3000,:] = util.img_as_float(I).flatten()
# Génération des labels d'apprentissage
y_train = np.concatenate((np.ones(3000), -np.ones(12000)))
# Apprentissage d'un SVM linéaire
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train, y_train)
# Chargement des images de test
X_test = np.zeros((1000+5256, 576))
for i in range(1000):
    I = io.imread('imageface/test/pos/%05d.png'%(i+1))
    X_test[i,:] = util.img_as_float(I).flatten()
for i in range(5256):
    I = io.imread('imageface/test/neg/%05d.png'%(i+1))
    X_test[i+1000,:] = util.img_as_float(I).flatten()
# Génération des labels de test
y_test = np.concatenate((np.ones(1000), -np.ones(5256)))
# Prédiction des scores sur les données de test
s_pred = clf.decision_function(X_test)
# Indice de tri décroissant des scores de prédiction
idx = np.argsort(s_pred)
idx = idx[::-1]
# Initialisation de la précision, du rappel et de VP, FP et FN
prec = np.zeros(len(s_pred))
rapp = np.zeros(len(s_pred))
VP = 0
FP = 0
FN = np.sum(y_test > 0)
# Parcours des prédictions par ordre décroissant de score
for k, i in enumerate(idx):
    # On ajoute s_pred[i] à la liste des détections de visage
    if y_test[i] > 0: # l'image correspond effectivement à un visage
        VP += 1
        FN -= 1
    else: # l'image ne correspond pas à un visage
        FP += 1
    prec[k] = VP/(VP+FP)
    rapp[k] = VP/(VP+FN)
    
# Courbe précision/rappel
import matplotlib.pyplot as plt
plt.plot(rapp, prec)
plt.xlim((0,1))
plt.ylim((0,1))
plt.xlabel('Rappel')
plt.ylabel('Précision')
plt.show()