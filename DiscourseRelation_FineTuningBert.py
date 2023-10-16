#!/usr/bin/env python
# coding: utf-8

# In[1]:


# A FAIRE

# Une classe pour le traitement des données (PDTB (pour gérer les différents splits ..) et SNLI)
# factoriser tout ce qui peut l'etre

# optimiser le max_length du tokenizer (est ce qu'il n'y a pas trop de padding?)
# mettre des "with torch.no_grad" la ou on peut (predict et evaluation ?)

# faire une methode pour les heatmap

# Commenter un peu la fin ...


# In[38]:


# Installations & Imports

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


pdtb2 = []
reader = csv.DictReader(open('pdtb2.csv', 'r'))
for example in reader:
    pdtb2.append(example)


# In[3]:


cb_split_set2sec = {"dev": [0, 1, 23, 24], "test": [21, 22], "train": list(range(2, 21))}
cb_split_sec2set = {}
for key, value in cb_split_set2sec.items():
    for section in value:
        cb_split_sec2set[section] = key
print(cb_split_set2sec)
print(cb_split_sec2set)


# In[4]:


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)


# In[31]:


# Chargement des fichiers pdtb2 test/train/dev vectorisés
# Arg1 contient les 1eres phrases, Arg2 les deuxiemes, y les goldclass

Arg1, Arg2 = defaultdict(lambda:[]), defaultdict(lambda:[])
X, y = defaultdict(lambda:[]), defaultdict(lambda:[])
shuffle(pdtb2)
for example in pdtb2:
    if example['Relation'] == 'Implicit':
        Arg1[cb_split_sec2set[int(example['Section'])]].append(example['Arg1_RawText'])
        Arg2[cb_split_sec2set[int(example['Section'])]].append(example['Arg2_RawText'])
        y[cb_split_sec2set[int(example['Section'])]].append(example['ConnHeadSemClass1'].split('.')[0])

print(Arg1['train'][0], Arg2['train'][0], y['train'][0])


# In[69]:


# nombre d'exemples par set
print(len(Arg1['train']), len(Arg2['train']), len(y['train']))
print(len(Arg1['dev']), len(Arg2['dev']), len(y['dev']))
print(len(Arg1['test']), len(Arg2['test']), len(y['test']))


# In[70]:


# distribution des labels
print("train :", Counter(y['train']))
print("dev :", Counter(y['dev']))
print("test :", Counter(y['test']))


# In[10]:


# création d'une correspondance (goldclass <-> entier) à l'aide :
# - d'une liste i2gold_class qui a un entier associe une class
# - d'un dictionnaire gold_class2i qui a une classe associe un entier

i2gold_class = list(set(y['train']))
gold_class2i = {gold_class : i for i, gold_class in enumerate(i2gold_class)}
print(i2gold_class)
print(gold_class2i)


# In[58]:


# on remplace les gold_class par les entiers associés dans y
# (pour pouvoir le tensoriser par la suite)
for s in ['test', 'train', 'dev']:
    y[s] = [gold_class2i[gold_class] for gold_class in y[s]]


# In[62]:


# un MLP qui prend en entrée une phrase vectorisée et renvoie la liste des probas pour certaines classes du PDTB2
# (à voir quelles classes nous intéressent le plus ..)

# Pour l'instant :
#   2 couches cachées
#   relu en premiere fonction d'activation
#   tanh en deuxieme fonction d'activation
#   du dropout à chaque couche (sauf à l'entrée)


class BertMLP(nn.Module):

    def __init__(self, first_hidden_layer_size, second_hidden_layer_size, size_of_batch, dropout, size_of_input=768, num_classes=len(i2gold_class), reg=50, loss=nn.NLLLoss()):
          
        super(BertMLP, self).__init__()

        self.reg = reg
        
        self.loss = loss

        self.size_of_batch = size_of_batch

        self.w1 = nn.Linear(size_of_input, first_hidden_layer_size)
        self.w2 = nn.Linear(first_hidden_layer_size, second_hidden_layer_size)
        self.w3 = nn.Linear(second_hidden_layer_size, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):

        """print(bert_model(**tokens))
        print(len(bert_model(**tokens)))
        print(bert_model(**tokens)[0].shape)
        print(bert_model(**tokens)[1].shape)"""
        vect_sentences = bert_model(**tokens)[0][:, 0, :]
        
        linear_comb = self.w1(vect_sentences)
        drop = self.dropout(linear_comb)
        out = torch.relu(drop)
    
        linear_comb = self.w2(out)
        drop = self.dropout(linear_comb)
        out = torch.tanh(drop)
    
        linear_comb = self.w3(out)
        drop = self.dropout(linear_comb)
        log_prob = F.log_softmax(drop, dim=1)

        return log_prob

    # l'entrainement du MLP

    # reg : regularite du calcul de la loss (on calcule la loss toutes les reg epoques)
    # down_sampling : booleen pour savoir si on fait du down sampling
    # size_of_samples : taille des samples lorsqu'on fait du down sampling

    def training_step(self, optimizer, nb_epoch = 50000, patience = 3, reg = 20, down_sampling = True, size_of_samples = 800):
        # les listes qui contiendront les valeurs de la loss sur le dev et le train pour chaque époque
        dev_losses = []
        train_losses = []

        for epoch in range(nb_epoch) :

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

            else: arg1_sample, arg2_sample, y_sample = Arg1['train'], Arg2['train'], y['train']

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
                arg1, arg2, gold_classes = arg1_sample[i: i+self.size_of_batch], arg2_sample[i: i+self.size_of_batch], torch.LongTensor(y_sample[i: i+self.size_of_batch])

                tokens = tokenizer(arg1, arg2, truncation=True, max_length=512,
                                   return_tensors="pt", padding='max_length')

                i += discourse_relation_mlp.size_of_batch

                optimizer.zero_grad()

                # calcul (du log) des probabilités de chaque classe, pour les exemples du batch
                log_probs = self.forward(tokens)

                # calcul de la loss
                loss = self.loss(log_probs, gold_classes)

                # MAJ des parametres
                loss.backward()
                optimizer.step()

            # print régulier pour savoir ou on en est
            # (ça peut être un peu long ..)
            if epoch % reg == 0:
                # mode "eval", pas de dropout ici
                self.eval()

                # on regarde la loss sur le dev et sur le train
                # pour pouvoir comparer l'évolution des 2

                # evaluation sur le dev
                tokens_dev = tokenizer(Arg1['dev'], Arg2['dev'], truncation=True, max_length = 512,
                                       return_tensors="pt", padding='max_length')
                log_probs_dev = self.forward(tokens_dev)
                loss_on_dev = self.loss(log_probs_dev, torch.LongTensor(y['dev'])).detach().numpy()
                dev_losses.append(loss_on_dev)

                # evaluation sur le train
                tokens_train = tokenizer(Arg1['train'], Arg2['train'], truncation=True, max_length = 512,
                                         return_tensors="pt", padding='max_length')
                log_probs_train = self.forward(tokens_train)
                loss_on_train = self.loss(log_probs_train, torch.LongTensor(y['train'])).detach().numpy()
                train_losses.append(loss_on_train)

                # early stopping
                if patience < len(dev_losses) and all([dev_losses[-i-1]<dev_losses[-i] and train_losses[-i]<train_losses[-i-1] for i in range(1, patience+1)]):
                    print(f"Epoch {epoch}\nloss on train is {loss_on_train}\nloss on dev is {loss_on_dev}\n")
                    print("EARLY STOPPING")
                    return dev_losses, train_losses

                print(f"Epoch {epoch}\nloss on train is {loss_on_train}\nloss on dev is {loss_on_dev}\n")

        return dev_losses, train_losses
      
    def predict(self, arg1, arg2):
        i = 0
        predictions = torch.tensor([])

        while i < len(arg1):
            tokens = tokenizer(arg1[i:i+self.size_of_batch], arg2[i:i+self.size_of_batch], truncation=True, max_length = 512, return_tensors="pt", padding='max_length')
            log_probs = self.forward(tokens)
            i += self.size_of_batch
            predictions = torch.cat((predictions, torch.argmax(log_probs, dim=1)))
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
discourse_relation_mlp = BertMLP(first_hidden_layer_size=400, second_hidden_layer_size=200, size_of_batch=10, dropout=0.3, loss=nn.NLLLoss())

# quelques hyperparametres
learning_rate = 0.0001
l2_reg = 0.0001

# choix de l'optimizer (SGD, Adam, Autre ?)
optim = torch.optim.Adam(discourse_relation_mlp.parameters(), lr=learning_rate, weight_decay=l2_reg)

dev_losses, train_losses = discourse_relation_mlp.training_step(optimizer=optim)


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
abs = list(range(0, len(dev_losses)*discourse_relation_mlp.reg, discourse_relation_mlp.reg))
plt.plot(abs, dev_losses, label='loss on dev set')
plt.plot(abs, train_losses, label='loss on train set')
plt.ylabel('loss')
plt.xlabel('nombre d\'époques')
plt.legend()
plt.show()
plt.savefig('BertFineTunedModel.png')


# In[ ]:


# sauvegarde d'un modele
torch.save(discourse_relation_mlp, 'BertFineTuned_model.pth')

# chargement d'un modele
# discourse_relation_mlp = torch.load('fourth_model.pth')


# In[32]:


# ouverture du csv associé au test du SNLI
snli_test = pd.read_csv("datas/snli_1.0/snli_1.0/snli_1.0_test.txt", sep="\t")

# les colonnes qui nous intéressent : 
#   les 2 phrases (dans l'ordre)
#   la goldclass
snli_test = snli_test[['gold_label','sentence1','sentence2']]

Arg1_nli = []
Arg2_nli = []
y_nli = []
for gold, sent1, sent2 in zip(snli_test['gold_label'], snli_test['sentence1'], snli_test['sentence2']):
    if gold != '-':
        if isinstance(sent2, float):
            print(sent1, '\n', sent2, '\n', gold, '\n')
        else:
            Arg1_nli.append(sent1)
            Arg2_nli.append(sent2)
            y_nli.append(gold)
            
print(Arg1_nli[0], Arg2_nli[0], y_nli[0])


# In[42]:


predict_NLI = discourse_relation_mlp.predict(Arg1_nli, Arg2_nli)
print(predict_NLI)


# In[43]:


repartition = Counter([(nli_class, i2gold_class[disc_rel]) for nli_class, disc_rel in zip(y_nli, predict_NLI)])
print(repartition)


# In[40]:


print(Counter(y_nli))


# In[54]:


i2nli = ['contradiction', 'entailment', 'neutral']

mat = torch.tensor([[repartition[(nli_class, rel)] for rel in i2gold_class] for nli_class in i2nli])
print(mat)

mat1 = mat.T/torch.sum(mat, axis=1)
print(mat1)

mat2 = mat/torch.sum(mat, axis=0)
print(mat2.T)


# In[53]:


df_cm = DataFrame(mat.T, index=i2gold_class, columns=['contradiction', 'entailment', 'neutral'])
ax = sn.heatmap(df_cm, cmap='Blues')

figure = ax.get_figure()
figure.savefig('AvantNormalisation.png', dpi=400)


# In[50]:


df_cm = DataFrame(mat1, index=i2gold_class, columns=['contradiction', 'entailment', 'neutral'])
ax = sn.heatmap(df_cm, cmap='Blues')

figure = ax.get_figure()
figure.savefig('ApresNormalisationSNLI.png', dpi=400)


# In[51]:


df_cm = DataFrame(mat2.T, index=i2gold_class, columns=['contradiction', 'entailment', 'neutral'])
ax = sn.heatmap(df_cm, cmap='Blues')

figure = ax.get_figure()
figure.savefig('ApresNormalisationPDTB.png', dpi=400)

