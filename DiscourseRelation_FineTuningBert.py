"""
A FAIRE

UN ARGPARSE

Une classe pour le traitement des données (PDTB (qui pourra notamment gérer les differents splits..) et SNLI)
factoriser tout ce qui peut l'etre

mettre des "with torch.no_grad" la ou c'est possible

faire une methode pour les heatmap

Commenter un peu la fin ...
"""
# --------------------- Installations et Imports -------------------------
import csv
from collections import Counter, defaultdict
from MyBertMLP import BertMLP
import pandas as pd
from pandas import DataFrame
import seaborn as sn
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

MAX_LENGTH = 128

pdtb2 = []
reader = csv.DictReader(open('datas/pdtb2.csv/pdtb2.csv', 'r'))
for example in reader:
    pdtb2.append(example)


# -------------------- Split ------------------------------

cb_split_set2sec = {"dev": [0, 1, 23, 24], "test": [21, 22], "train": list(range(2, 21))}
cb_split_sec2set = {}
for key, value in cb_split_set2sec.items():
    for section in value:
        cb_split_sec2set[section] = key


"""
Chargement du fichier pdtb2
Arg1 contient les 1eres phrases, Arg2 les deuxiemes, y les goldclass
"""

Arg1, Arg2 = defaultdict(lambda: []), defaultdict(lambda: [])
X, y = defaultdict(lambda: []), defaultdict(lambda: [])
# shuffle(pdtb2)
for example in pdtb2:
    if example['Relation'] == 'Implicit':
        Arg1[cb_split_sec2set[int(example['Section'])]].append(example['Arg1_RawText'])
        Arg2[cb_split_sec2set[int(example['Section'])]].append(example['Arg2_RawText'])
        y[cb_split_sec2set[int(example['Section'])]].append(example['ConnHeadSemClass1'].split('.')[0])


# -------------------- ouverture du csv associé au test du SNLI ------------------------------
"""
les colonnes qui nous intéressent :
    les 2 phrases (dans l'ordre)
    la goldclass
"""

snli_test = pd.read_csv("datas/snli_1.0/snli_1.0/snli_1.0_test.txt", sep="\t")
snli_test = snli_test[['gold_label', 'sentence1', 'sentence2']]

y_nli = []
for gold, sent1, sent2 in zip(snli_test['gold_label'], snli_test['sentence1'], snli_test['sentence2']):
    if gold != '-':
        if isinstance(sent2, float):
            print(sent1, '\n', sent2, '\n', gold, '\n')
        else:
            Arg1['snli test'].append(sent1)
            Arg2['snli test'].append(sent2)
            y_nli.append(gold)


# nombre d'exemples par set
# (On compare egalement qu'il y a bien autant de Arg1 que de Arg2)
"""
print(len(Arg1['train']), len(Arg2['train']), len(y['train']))
print(len(Arg1['dev']), len(Arg2['dev']), len(y['dev']))
print(len(Arg1['test']), len(Arg2['test']), len(y['test']))
"""

# distribution des labels
"""
print("train :", Counter(y['train']))
print("dev :", Counter(y['dev']))
print("test :", Counter(y['test']))
"""

# création d'une correspondance (goldclass <-> entier) à l'aide :
# - d'une liste i2gold_class qui a un entier associe une class
# - d'un dictionnaire gold_class2i qui a une classe associe un entier

i2gold_class = list(set(y['train']))
gold_class2i = {gold_class : i for i, gold_class in enumerate(i2gold_class)}
print(i2gold_class)
print(gold_class2i)

# on remplace les gold_class par les entiers associés dans y
# (pour pouvoir le tensoriser par la suite)
for s in ['test', 'train', 'dev']:
    y[s] = [gold_class2i[gold_class] for gold_class in y[s]]


# --------------------- création du classifieur -----------------------

discourse_relation_mlp = BertMLP(first_hidden_layer_size=50, second_hidden_layer_size=50, size_of_batch=100,
                                 dropout=0.5, loss=nn.NLLLoss(), device=device, num_classes=len(i2gold_class),
                                 Arg1train=Arg1['train'], Arg2train=Arg2['train'], ytrain=y['train'],
                                 Arg1dev=Arg1['dev'], Arg2dev=Arg2['dev'], ydev=y['dev'],
                                 i2goldclasses=i2gold_class)
discourse_relation_mlp = discourse_relation_mlp.to(device)

# quelques hyperparametres
learning_rate = 0.00001
l2_reg = 0.0002

# choix de l'optimizer (SGD, Adam, Autre ?)
optim = torch.optim.Adam(discourse_relation_mlp.parameters(),
                         lr=learning_rate,
                         weight_decay=l2_reg)

dev_losses, train_losses = discourse_relation_mlp.training_step(optimizer=optim,
                                                                nb_epoch=1,
                                                                down_sampling=False)


predict_train = discourse_relation_mlp.predict(Arg1['train'], Arg2['train'])
print(i2gold_class)


discourse_relation_mlp.evaluation("train", Arg1["train"], Arg2["train"], y["train"])
discourse_relation_mlp.evaluation("dev", Arg1["dev"], Arg2["dev"], y["dev"])
discourse_relation_mlp.evaluation("test", Arg1["test"], Arg2["test"], y["test"])


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
loss_fig.savefig('Images/BertFineTunedModel.png')


# sauvegarde d'un modele
torch.save(discourse_relation_mlp, 'BertFineTuned_model.pth')

# chargement d'un modele
# discourse_relation_mlp = torch.load('fourth_model.pth')


# --------------- prediction des relations de discours sur le SNLI -----------------
predict_NLI = discourse_relation_mlp.predict(Arg1['snli test'], Arg2['snli test'])
predict_revNLI = discourse_relation_mlp.predict(Arg2['snli test'], Arg1['snli test'])


# ------------- La repartition des relations de discours predites sur le SNLI --------------------

repartition = Counter([(nli_class, i2gold_class[int(disc_rel)])
                       for nli_class, disc_rel in zip(y_nli, predict_NLI.tolist())])
repartition_rev = Counter([(nli_class, i2gold_class[int(disc_rel)])
                           for nli_class, disc_rel in zip(y_nli, predict_revNLI.tolist())])
# print(repartition)

i2nli = ['contradiction', 'entailment', 'neutral']


def save_plot(matrix, filename, index=i2gold_class, columns=['contradiction', 'entailment', 'neutral']):
    plt.figure()
    df_cm = DataFrame(matrix, index=index, columns=columns)
    ax = sn.heatmap(df_cm, cmap='Blues')
    heatmap = ax.get_figure()
    heatmap.savefig(filename, dpi=400)


mat = torch.tensor([[repartition[(nli_class, rel)] for rel in i2gold_class] for nli_class in i2nli])
save_plot(mat.T, 'Images/AvantNormalisation.png')

mat1 = mat.T/torch.sum(mat, axis=1)
save_plot(mat1, 'Images/ApresNormalisationSNLI.png')

mat2 = mat/torch.sum(mat, axis=0)
save_plot(mat2.T, 'Images/ApresNormalisationPDTB.png')


mat = torch.tensor([[repartition_rev[(nli_class, rel)] for rel in i2gold_class] for nli_class in i2nli])
save_plot(mat.T, 'Images/AvantNormalisation_rev.png')

mat1 = mat.T/torch.sum(mat, axis=1)
save_plot(mat1, 'Images/ApresNormalisationSNLI_rev.png')

mat2 = mat/torch.sum(mat, axis=0)
save_plot(mat2.T, 'Images/ApresNormalisationPDTB_rev.png')