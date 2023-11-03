"""
A FAIRE :

UN ARGPARSE

Une classe pour le traitement des données (PDTB (qui pourra notamment gérer les differents splits..) et SNLI)
factoriser tout ce qui peut l'etre

mettre des "with torch.no_grad" la ou c'est possible

faire une methode pour les heatmap

Commenter un peu la fin ...
"""
from random import shuffle

# --------------------- Installations et Imports -------------------------

from MyBertMLP import BertMLP
from DatasGeneration import PDTBReader, SNLIReader

import csv
from collections import Counter, defaultdict

import pandas as pd
from pandas import DataFrame
import seaborn as sn
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import os
import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

MAX_LENGTH = 128


# ------------------------ Lecture des fichiers ------------------------
relation = 'Implicit'
pdtb = PDTBReader()
pdtb.read(split='Ji', relation='Implicit')

Arg1PDTB, Arg2PDTB, conn, rel = pdtb.Arg1, pdtb.Arg2, pdtb.conn, pdtb.rel

snli = SNLIReader()
snli.read(part='dev')

Arg1SNLI, Arg2SNLI, y = snli.Arg1, snli.Arg2, snli.y

# nombre d'exemples par set
# (On compare egalement qu'il y a bien autant de Arg1 que de Arg2)
"""
print(len(Arg1['train']), len(Arg2['train']), len(y['train']))
print(len(Arg1['dev']), len(Arg2['dev']), len(y['dev']))
print(len(Arg1['test']), len(Arg2['test']), len(y['test']))
"""

# distribution des labels

print(relation, " train :", Counter(rel[relation + '_train']))
print(relation, " train :", Counter(conn[relation + '_train']))

print(relation, " dev :", Counter(rel[relation + '_dev']))
print(relation, " dev :", Counter(conn[relation + '_dev']))

print(relation, " test :", Counter(rel[relation + '_test']))
print(relation, " test :", Counter(conn[relation + '_test']))


# création d'une correspondance (goldclass <-> entier) à l'aide :
# - d'une liste i2gold_class qui a un entier associe une class
# - d'un dictionnaire gold_class2i qui a une classe associe un entier

i2gold_class = list(set(rel[relation + '_train']))
gold_class2i = {gold_class: i for i, gold_class in enumerate(i2gold_class)}
print(i2gold_class)
print(gold_class2i)

# on remplace les gold_class par les entiers associés dans y
# (pour pouvoir le tensoriser par la suite)
for s in [relation + '_test', relation + '_train', relation + '_dev']:
    rel[s] = [gold_class2i[gold_class] for gold_class in rel[s]]


# -------------------------- création du classifieur -------------------------------

discourse_relation_mlp = BertMLP(first_hidden_layer_size=50, second_hidden_layer_size=25, size_of_batch=100,
                                 dropout=0.6, loss=nn.NLLLoss(), device=device, num_classes=len(i2gold_class),
                                 Arg1train=Arg1PDTB[relation + '_train'], Arg2train=Arg2PDTB[relation + '_train'],
                                 ytrain=rel[relation + '_train'],
                                 Arg1dev=Arg1PDTB[relation + '_dev'], Arg2dev=Arg2PDTB[relation + '_dev'],
                                 ydev=rel[relation + '_dev'],
                                 i2goldclasses=i2gold_class)

discourse_relation_mlp = discourse_relation_mlp.to(device)


# choix de l'optimizer (SGD, Adam, Autre ?)
optim = torch.optim.Adam(discourse_relation_mlp.parameters(),
                         lr=0.0001,
                         weight_decay=0.0002)

# entrainement
dev_losses, train_losses = discourse_relation_mlp.training_step(optimizer=optim,
                                                                nb_epoch=1,
                                                                patience=2,
                                                                down_sampling=True,
                                                                size_of_samples=3000,
                                                                fixed_sampling=False)


discourse_relation_mlp.evaluation("train", Arg1PDTB[relation + '_train'],
                                  Arg2PDTB[relation + '_train'], rel[relation + '_train'])
discourse_relation_mlp.evaluation("dev", Arg1PDTB[relation + '_dev'],
                                  Arg2PDTB[relation + '_dev'], rel[relation + '_dev'])
discourse_relation_mlp.evaluation("test", Arg1PDTB[relation + '_test'],
                                  Arg2PDTB[relation + '_test'], rel[relation + '_test'])


# courbe d'evolution de la loss

def plot_loss():
    plt.figure()
    abs = list(range(0, len(dev_losses)*discourse_relation_mlp.reg, discourse_relation_mlp.reg))
    loss_fig = plt.figure("Figure 1")
    plt.plot(abs, dev_losses, label='loss on dev set')
    plt.plot(abs, train_losses, label='loss on train set')
    plt.ylabel('loss')
    plt.xlabel('nombre d\'époques')
    plt.legend()
    return loss_fig


loss_fig = plot_loss()
loss_fig.savefig(os.path.join(".", "Images", 'BertFineTunedModel.png'))


# sauvegarde d'un modele
torch.save(discourse_relation_mlp, 'BertFineTuned_model.pth')

# chargement d'un modele
# discourse_relation_mlp = torch.load('fourth_model.pth')


# --------------- prediction des relations de discours sur le SNLI -----------------------------
# ordre P H:
predict_NLI = discourse_relation_mlp.predict(Arg1SNLI['dev'], Arg2SNLI['dev'])

# ordre H P :
predict_revNLI = discourse_relation_mlp.predict(Arg2SNLI['dev'], Arg1SNLI['dev'])


# ------------- La repartition des relations de discours predites sur le SNLI --------------------

repartition = Counter([(nli_class, i2gold_class[int(disc_rel)])
                       for nli_class, disc_rel in zip(y['dev'], predict_NLI.tolist())])
repartition_rev = Counter([(nli_class, i2gold_class[int(disc_rel)])
                           for nli_class, disc_rel in zip(y['dev'], predict_revNLI.tolist())])

comb = Counter([(nli_class, (i2gold_class[int(disc_rel)], i2gold_class[int(disc_rel_rev)]))
                for nli_class, disc_rel, disc_rel_rev in zip(y['dev'], predict_NLI.tolist(), predict_revNLI.tolist())])

i2nli = ['contradiction', 'entailment', 'neutral']


def save_plot(matrix, filename, index=i2gold_class, columns=i2nli):
    plt.figure()
    df_cm = DataFrame(matrix, index=index, columns=columns)
    ax = sn.heatmap(df_cm, cmap='Blues', annot=True, fmt=".2f")
    heatmap = ax.get_figure()
    heatmap.savefig(filename, bbox_inches='tight')


withoutnorm = torch.tensor([[repartition[(nli_class, rel)] for rel in i2gold_class] for nli_class in i2nli])
save_plot(withoutnorm.T, os.path.join(".", "Images", 'AvantNormalisation.png'))

snlinorm = withoutnorm.T/torch.sum(withoutnorm, axis=1)
save_plot(snlinorm, os.path.join(".", "Images", 'ApresNormalisationSNLI.png'))

pdtbnorm = withoutnorm/torch.sum(withoutnorm, axis=0)
save_plot(pdtbnorm.T, os.path.join(".", "Images", 'ApresNormalisationPDTB.png'))


mat = torch.tensor([[repartition_rev[(nli_class, rel)] for rel in i2gold_class] for nli_class in i2nli])
save_plot(mat.T, os.path.join(".", "Images", 'AvantNormalisation_rev.png'))

mat1 = mat.T/torch.sum(mat, axis=1)
save_plot(mat1, os.path.join(".", "Images", 'ApresNormalisationSNLI_rev.png'))

mat2 = mat/torch.sum(mat, axis=0)
save_plot(mat2.T, os.path.join(".", "Images", 'ApresNormalisationPDTB_rev.png'))


i2gold_class_squared = [(rel1, rel2) for rel1 in i2gold_class for rel2 in i2gold_class]

mat = torch.tensor([[comb[(nli_class, rel_couple)] for rel_couple in i2gold_class_squared] for nli_class in i2nli])
save_plot(mat.T, os.path.join(".", "Images", 'AvantNormalisation_comb.png'), index=i2gold_class_squared)

mat1 = mat.T/torch.sum(mat, axis=1)
save_plot(mat1, os.path.join(".", "Images", 'ApresNormalisationSNLI_comb.png'), index=i2gold_class_squared)

mat2 = mat/torch.sum(mat, axis=0)
save_plot(mat2.T, os.path.join(".", "Images", 'ApresNormalisationPDTB_comb.png'), index=i2gold_class_squared)


zipped = list(zip(Arg1SNLI['dev'], Arg2SNLI['dev'], y['dev'],
                  predict_NLI.tolist(), predict_revNLI.tolist()))
shuffle(zipped)
Args1, Args2, y, predict_NLI, predict_revNLI = zip(*zipped)

compteur = defaultdict(lambda: 0)
with open('examples.txt', 'w') as f:
    for arg1, arg2, nli_class, rel, rel_rev in zip(Args1, Args2, y,
                                                   predict_NLI,
                                                   predict_revNLI):
        rel = i2gold_class[int(rel)]
        rel_rev = i2gold_class[int(rel_rev)]
        if compteur[rel + rel_rev + nli_class] == 5:
            pass
        else:
            compteur[rel + rel_rev + nli_class] += 1
            f.write('Classe NLI : ' + nli_class + '\n')
            f.write('Rel(Arg1, Arg2) : ' + rel + '\n')
            f.write('Rel(Arg2, Arg1) : ' + rel_rev + '\n')
            f.write('Arg1 : ' + arg1 + '\n')
            f.write('Arg2 : ' + arg2 + '\n')
            f.write('\n')
