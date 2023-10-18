#!/usr/bin/env python
# coding: utf-8
"""
A FAIRE

Une classe pour le traitement des données (PDTB (qui pourra notamment gérer les differents splits..) et SNLI)
factoriser tout ce qui peut l'etre

mettre des "with torch.no_grad" la ou c'est possible

faire une methode pour les heatmap

Commenter un peu la fin ...
"""


"""
--------------------- Installations et Imports -------------------------
"""
import csv

from transformers import AutoModel, AutoTokenizer
from random import shuffle
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score

import random
import pandas as pd
from pandas import DataFrame
import seaborn as sn
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle


# In[2]:

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

MAX_LENGTH = 128

pdtb2 = []
reader = csv.DictReader(open('pdtb2.csv', 'r'))
for example in reader:
    pdtb2.append(example)


# In[3]:
"""
-------------------- Split ------------------------------
"""
cb_split_set2sec = {"dev": [0, 1, 23, 24], "test": [21, 22], "train": list(range(2, 21))}
cb_split_sec2set = {}
for key, value in cb_split_set2sec.items():
    for section in value:
        cb_split_sec2set[section] = key

"""
print(cb_split_set2sec)
print(cb_split_sec2set)
"""

# In[4]:
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name).to(device)


# In[31]:
"""
Chargement du fichier pdtb2
Arg1 contient les 1eres phrases, Arg2 les deuxiemes, y les goldclass
"""

Arg1, Arg2 = defaultdict(lambda: []), defaultdict(lambda: [])
X, y = defaultdict(lambda: []), defaultdict(lambda: [])
shuffle(pdtb2)
for example in pdtb2[:300]:
    if example['Relation'] == 'Implicit':
        Arg1[cb_split_sec2set[int(example['Section'])]].append(example['Arg1_RawText'])
        Arg2[cb_split_sec2set[int(example['Section'])]].append(example['Arg2_RawText'])
        y[cb_split_sec2set[int(example['Section'])]].append(example['ConnHeadSemClass1'].split('.')[0])

#print(Arg1['train'][0], Arg2['train'][0], y['train'][0])


# In[32]:
"""
-------------------- ouverture du csv associé au test du SNLI ------------------------------

les colonnes qui nous intéressent :
    les 2 phrases (dans l'ordre)
    la goldclass
"""

snli_test = pd.read_csv("datas/snli_1.0/snli_1.0/snli_1.0_test.txt", sep="\t")

snli_test = snli_test[['gold_label', 'sentence1', 'sentence2']]

y_nli = []
for gold, sent1, sent2 in zip(snli_test['gold_label'], snli_test['sentence1'], snli_test['sentence2']):
    if gold != '-' and len(y_nli) < 200:
        if isinstance(sent2, float):
            print(sent1, '\n', sent2, '\n', gold, '\n')
        else:
            Arg1['snli test'].append(sent1)
            Arg2['snli test'].append(sent2)
            y_nli.append(gold)


# nombre d'exemples par set
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


# In[10]:
# tokenisation des sets

tokenized = {'dev': tokenizer(Arg1['dev'], Arg2['dev'], truncation=True, max_length=MAX_LENGTH,
                              return_tensors="pt", padding='max_length'),
             'test': tokenizer(Arg1['test'], Arg2['test'], truncation=True, max_length=MAX_LENGTH,
                               return_tensors="pt", padding='max_length'),
             'train': tokenizer(Arg1['train'], Arg2['train'], truncation=True, max_length=MAX_LENGTH,
                                return_tensors="pt", padding='max_length'),
             'snli test': tokenizer(Arg1['snli test'], Arg2['snli test'], truncation=True, max_length=MAX_LENGTH,
                                    return_tensors="pt", padding='max_length')
             }


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



"""
----------------------- Le Modèle ---------------------------------

un MLP qui prend en entrée une phrase tokenisée et renvoie la liste des probas pour certaines classes du PDTB2
(à voir quelles classes nous intéressent le plus ..)

"""


class BertMLP(nn.Module):

    def __init__(self, first_hidden_layer_size, size_of_batch, dropout, size_of_input=768,
                 num_tokens=MAX_LENGTH, num_classes=len(i2gold_class), reg=5, loss=nn.NLLLoss().to(device)):
          
        super(BertMLP, self).__init__()

        self.reg = reg
        
        self.loss = loss

        self.num_tokens = num_tokens
        self.size_of_batch = size_of_batch

        self.w1 = nn.Linear(size_of_input, first_hidden_layer_size).to(device)
        # self.w2 = nn.Linear(first_hidden_layer_size, second_hidden_layer_size)
        self.w3 = nn.Linear(first_hidden_layer_size, num_classes).to(device)

        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, tokens):

        vect_sentences = bert_model(**tokens.to(device))[0][:, 0, :].to(device)
        
        linear_comb = self.w1(vect_sentences).to(device)
        drop = self.dropout(linear_comb).to(device)
        out = torch.relu(drop).to(device)

        linear_comb = self.w3(out).to(device)
        drop = self.dropout(linear_comb).to(device)
        log_prob = F.log_softmax(drop, dim=1).to(device)

        return log_prob

    # l'entrainement du MLP

    # reg : regularite du calcul de la loss (on calcule la loss toutes les reg epoques)
    # down_sampling : booleen pour savoir si on fait du down sampling
    # size_of_samples : taille des samples lorsqu'on fait du down sampling

    def training_step(self, optimizer, nb_epoch=2, patience=2, reg=1, down_sampling=True, size_of_samples=800):
        # les listes qui contiendront les valeurs de la loss sur le dev et le train pour chaque époque
        dev_losses = []
        train_losses = []

        for epoch in range(nb_epoch):

            # est-ce qu'on veut faire du downsampling ou non
            if down_sampling:
                # creation du sample sur lequel on va s'entrainer
                arg1_sample = []
                arg2_sample = []
                y_sample = []

                for gold_class in i2gold_class:

                    # shuffle pour ne pas prendre les mêmes exemples à chaque fois
                    shuffle(examples[gold_class])

                    for vector in examples[label][:size_of_samples]:
                        arg1_sample.append(vector[0])
                        arg2_sample.append(vector[1])
                        y_sample.append(label)

            else:
                arg1_sample, arg2_sample, y_sample = Arg1['train'], Arg2['train'], y['train']

            # melange du sample pour ne pas toujours s'entrainer sur les labels dans le même ordre
            sample = list(zip(arg1_sample, arg2_sample, y_sample))
            shuffle(sample)
            arg1_sample, arg2_sample, y_sample = zip(*sample)

            # mode "train", on précise quand on s'entraine et quand on évalue le modèle pour le dropout
            self.train()

            # un indice i pour parcourir tout le sample
            i = 0
            while i < len(y_sample):
                # input_vectors : les vecteurs contenant les ids et les masks pour chaque exemple
                # gold_classes : les goldclass associées
                arg1, arg2 = arg1_sample[i: i+self.size_of_batch], arg2_sample[i: i+self.size_of_batch]
                gold_classes = torch.LongTensor(y_sample[i: i+self.size_of_batch]).to(device)

                tokens = tokenizer(arg1, arg2, truncation=True, max_length=self.num_tokens,
                                   return_tensors="pt", padding='max_length')

                i += self.size_of_batch

                optimizer.zero_grad()

                # calcul (du log) des probabilités de chaque classe, pour les exemples du batch
                log_probs = self.forward(tokens.to(device)).to(device)

                # calcul de la loss
                loss = self.loss(log_probs, gold_classes)

                # MAJ des parametres
                loss.backward()
                optimizer.step()

            # print régulier de la loss sur le train et le dev
            if epoch % reg == 0:
                # mode "eval", pas de dropout ici
                self.eval()

                # on regarde la loss sur le dev et sur le train
                # pour pouvoir comparer l'évolution des 2
                with torch.no_grad():

                    # evaluation sur le dev
                    log_probs_dev = self.forward(tokenized['dev'])
                    loss_on_dev = self.loss(log_probs_dev, torch.LongTensor(y['dev']).to(device)).cpu().detach().numpy()
                    dev_losses.append(loss_on_dev)

                    # evaluation sur le train
                    log_probs_train = self.forward(tokenized['train'])
                    loss_on_train = self.loss(log_probs_train, torch.LongTensor(y['train']).to(device)).cpu().detach().numpy()
                    train_losses.append(loss_on_train)

                # early stopping
                if patience < len(dev_losses) and all([dev_losses[-i-1]<dev_losses[-i] and train_losses[-i]<train_losses[-i-1] for i in range(1, patience+1)]):
                    print(f"Epoch {epoch}\nloss on train is {loss_on_train}\nloss on dev is {loss_on_dev}\n")
                    print("EARLY STOPPING")
                    return dev_losses, train_losses

                print(f"Epoch {epoch}\nloss on train is {loss_on_train}\nloss on dev is {loss_on_dev}\n")

        return dev_losses, train_losses
      
    def predict(self, sentences1, sentences2):
        self.eval()
        i = 0
        predictions = torch.tensor([])
        with torch.no_grad():
            while i < len(sentences1):
                arg1, arg2 = sentences1[i: i+self.size_of_batch], sentences2[i: i+self.size_of_batch]
                batch_tokens = tokenizer(arg1, arg2, truncation=True, max_length=self.num_tokens,
                                         return_tensors="pt", padding='max_length')
                log_probs = self.forward(batch_tokens.to(device))
                i += self.size_of_batch
                predictions = torch.cat((predictions.to(device), torch.argmax(log_probs, dim=1).to(device)))
        return predictions

    def evaluation(self, data_set):

        y_true = torch.tensor(y[data_set])
        y_pred = self.predict(Arg1[data_set], Arg2[data_set])

        torch.save(torch.tensor(confusion_matrix(y_true, y_pred)), data_set+'_confmat.pt')
        print(confusion_matrix(y_true, y_pred))

        torch.save(torch.tensor(f1_score(y_true, y_pred, average='macro')), data_set+'_f1macro.pt')
        print("f1 macro : ", f1_score(y_true, y_pred, average='macro'))

        torch.save(torch.tensor(precision_score(y_true, y_pred, average='macro')), data_set+'_precisionmacro.pt')
        print("precision macro : ", precision_score(y_true, y_pred, average='macro'))

        torch.save(torch.tensor(accuracy_score(y_true, y_pred)), data_set+'_accuracy.pt')
        print("exactitude : ", accuracy_score(y_true, y_pred))


# In[21]:


examples = defaultdict(lambda:[])

for arg1, arg2, label in zip(Arg1['train'], Arg2['train'], y['train']):
    examples[label].append((arg1, arg2))


# In[63]:
# création du classifieur
discourse_relation_mlp = BertMLP(first_hidden_layer_size=50, size_of_batch=100, dropout=0.3, loss=nn.NLLLoss())
discourse_relation_mlp = discourse_relation_mlp.to(device)

# quelques hyperparametres
learning_rate = 0.0001
l2_reg = 0.0001

# choix de l'optimizer (SGD, Adam, Autre ?)
optim = torch.optim.Adam(discourse_relation_mlp.parameters(), lr=learning_rate, weight_decay=l2_reg)

dev_losses, train_losses = discourse_relation_mlp.training_step(optimizer=optim, nb_epoch=10, down_sampling=False)


# In[64]:
predict_train = discourse_relation_mlp.predict(Arg1['train'], Arg2['train'])
print(predict_train)
print(i2gold_class)


# In[65]:
discourse_relation_mlp.evaluation("train")

# In[66]:
discourse_relation_mlp.evaluation("dev")

# In[67]:
discourse_relation_mlp.evaluation("test")

# In[29]:
# courbe d'evolution de la loss


def plot_loss():
    abs = list(range(0, len(dev_losses)*discourse_relation_mlp.reg, discourse_relation_mlp.reg))
    loss_fig = plt.figure("Figure 1")
    plt.plot(abs, dev_losses, label='loss on dev set')
    plt.plot(abs, train_losses, label='loss on train set')
    plt.ylabel('loss')
    plt.xlabel('nombre d\'époques')
    plt.legend()
    return loss_fig


loss_fig = plot_loss
loss_fig.savefig('BertFineTunedModel.png')


# In[ ]:
# sauvegarde d'un modele
torch.save(discourse_relation_mlp, 'BertFineTuned_model.pth')

# chargement d'un modele
# discourse_relation_mlp = torch.load('fourth_model.pth')

# In[42]:
predict_NLI = discourse_relation_mlp.predict(tokenized['snli test'])
# print(predict_NLI[:10])


# In[43]:
repartition = Counter([(nli_class, i2gold_class[int(disc_rel)]) for nli_class, disc_rel in zip(y_nli, predict_NLI.tolist())])
# print(repartition)


# In[40]:
# print(Counter(y_nli))

# In[54]:
i2nli = ['contradiction', 'entailment', 'neutral']


def plot_mat(matrix, index=i2gold_class, columns=['contradiction', 'entailment', 'neutral']):
    df_cm = DataFrame(matrix, index=index, columns=columns)
    ax = sn.heatmap(df_cm, cmap='Blues')
    heatmap = ax.get_figure()
    return heatmap


mat = torch.tensor([[repartition[(nli_class, rel)] for rel in i2gold_class] for nli_class in i2nli])
print(mat)
figure = plot_mat(mat.T)
figure.savefig('AvantNormalisation.png', dpi=400)

mat1 = mat.T/torch.sum(mat, axis=1)
print(mat1)
figure = plot_mat(mat1)
figure.savefig('ApresNormalisationSNLI.png', dpi=400)

mat2 = mat/torch.sum(mat, axis=0)
print(mat2.T)
figure = plot_mat(mat2.T)
figure.savefig('ApresNormalisationPDTB.png', dpi=400)