"""
A FAIRE

Une classe pour le traitement des données (PDTB (qui pourra notamment gérer les differents splits..) et SNLI)
factoriser tout ce qui peut l'etre

mettre des "with torch.no_grad" la ou c'est possible

faire une methode pour les heatmap

Commenter un peu la fin ...
"""
# --------------------- Installations et Imports -------------------------
from random import shuffle
import sys
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import shuffle
from collections import Counter, defaultdict
from transformers import AutoModel, AutoTokenizer

import csv
from collections import Counter, defaultdict

import pandas as pd
from pandas import DataFrame
import seaborn as sn
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ----------------------- Le Modèle ---------------------------------

"""
un MLP qui prend en entrée une phrase tokenisée et renvoie la liste des probas pour certaines classes du PDTB2
(à voir quelles classes nous intéressent le plus ..)

"""


class BertMLP(nn.Module):

    def __init__(self, first_hidden_layer_size, second_hidden_layer_size, size_of_batch, dropout, device, num_classes,
                 Arg1train, Arg2train, ytrain, Arg1dev, Arg2dev, ydev, i2goldclasses,
                 size_of_input=768, num_tokens=128, reg=5, loss=nn.NLLLoss(),
                 model_name="bert-base-uncased"):

        super(BertMLP, self).__init__()

        self.reg = reg

        self.loss = loss
        self.Arg1train, self.Arg2train, self.ytrain = Arg1train, Arg2train, ytrain
        self.Arg1dev, self.Arg2dev, self.ydev = Arg1dev, Arg2dev, ydev
        self.num_tokens = num_tokens
        self.size_of_batch = size_of_batch
        self.i2goldclasses = i2goldclasses

        self.w1 = nn.Linear(size_of_input, first_hidden_layer_size).to(device)
        self.w2 = nn.Linear(first_hidden_layer_size, second_hidden_layer_size)
        self.w3 = nn.Linear(second_hidden_layer_size, num_classes).to(device)

        self.dropout = nn.Dropout(dropout).to(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name).to(device)

    def forward(self, tokens):

        vect_sentences = self.bert_model(**tokens.to(self.device))[0][:, 0, :].to(self.device)

        linear_comb = self.w1(vect_sentences).to(self.device)
        drop = self.dropout(linear_comb).to(self.device)
        out = torch.relu(drop).to(self.device)

        linear_comb = self.w2(out).to(self.device)
        drop = self.dropout(linear_comb).to(self.device)
        out = torch.relu(drop).to(self.device)

        linear_comb = self.w3(out).to(self.device)
        drop = self.dropout(linear_comb).to(self.device)
        log_prob = F.log_softmax(drop, dim=1).to(self.device)

        return log_prob

    # l'entrainement du MLP

    # reg : regularite du calcul de la loss (on calcule la loss toutes les reg epoques)
    # down_sampling : booleen pour savoir si on fait du down sampling
    # size_of_samples : taille des samples lorsqu'on fait du down sampling

    def training_step(self, optimizer, nb_epoch=2, patience=2, reg=1, down_sampling=True, size_of_samples=1000):
        # les listes qui contiendront les valeurs de la loss sur le dev et le train pour chaque époque
        dev_losses = []
        train_losses = []

        if down_sampling:
            examples = defaultdict(lambda: [])
            for arg1, arg2, label in zip(self.Arg1train, self.Arg2train, self.ytrain):
                examples[label].append((arg1, arg2))

        for epoch in range(nb_epoch):

            # est-ce qu'on veut faire du downsampling ou non
            if down_sampling:
                # creation du sample sur lequel on va s'entrainer
                arg1_sample = []
                arg2_sample = []
                y_sample = []

                for gold_class in range(len(self.i2goldclasses)):
                    # shuffle pour ne pas prendre les mêmes exemples à chaque fois
                    shuffle(examples[gold_class])

                    for vector in examples[gold_class][:size_of_samples]:
                        arg1_sample.append(vector[0])
                        arg2_sample.append(vector[1])
                        y_sample.append(gold_class)

            else:
                arg1_sample, arg2_sample, y_sample = self.Arg1train, self.Arg2train, self.ytrain

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
                gold_classes = torch.LongTensor(y_sample[i: i+self.size_of_batch]).to(self.device)

                tokens = self.tokenizer(arg1, arg2, truncation=True, max_length=self.num_tokens,
                                        return_tensors="pt", padding='max_length')

                i += self.size_of_batch

                optimizer.zero_grad()

                # calcul (du log) des probabilités de chaque classe, pour les exemples du batch
                log_probs = self.forward(tokens.to(self.device)).to(self.device)

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
                    loss_on_dev = self.computing_loss(self.Arg1dev, self.Arg2dev, self.ydev)
                    dev_losses.append(loss_on_dev)

                    # evaluation sur le train
                    loss_on_train = self.computing_loss(self.Arg1train, self.Arg2train, self.ytrain)
                    train_losses.append(loss_on_train)

                # early stopping
                if (patience < len(dev_losses) and all([dev_losses[-i-1] < dev_losses[-i] and
                                                        train_losses[-i] < train_losses[-i-1] for i in range(1, patience + 1)])):
                    print(f"Epoch {epoch}\nloss on train is {loss_on_train}\nloss on dev is {loss_on_dev}\n")
                    print("EARLY STOPPING")
                    return dev_losses, train_losses

                print(f"Epoch {epoch}\nloss on train is {loss_on_train}\nloss on dev is {loss_on_dev}\n")

        return dev_losses, train_losses

    def computing_loss(self, arg1, arg2, y):
        self.eval()
        i, loss, nb_batch = 0, 0, 0
        with torch.no_grad():
            while i < len(arg1):
                arg1_sample, arg2_sample = arg1[i: i+self.size_of_batch], arg2[i: i+self.size_of_batch]
                batch_tokens = self.tokenizer(arg1_sample, arg2_sample, truncation=True, max_length=self.num_tokens,
                                              return_tensors="pt", padding='max_length')
                log_probs = self.forward(batch_tokens.to(self.device))
                loss += self.loss(log_probs, torch.LongTensor(y[i: i+self.size_of_batch]).to(self.device)).cpu().detach().numpy()
                i += self.size_of_batch
                nb_batch += 1
        return loss/nb_batch

    def predict(self, sentences1, sentences2):
        self.eval()
        i = 0
        predictions = torch.tensor([])
        with torch.no_grad():
            while i < len(sentences1):
                arg1, arg2 = sentences1[i: i+self.size_of_batch], sentences2[i: i+self.size_of_batch]
                batch_tokens = self.tokenizer(arg1, arg2, truncation=True, max_length=self.num_tokens,
                                              return_tensors="pt", padding='max_length')
                log_probs = self.forward(batch_tokens.to(self.device))
                i += self.size_of_batch
                predictions = torch.cat((predictions.to(self.device), torch.argmax(log_probs, dim=1).to(self.device)))
        return predictions

    def evaluation(self, data_set, arg1, arg2, y):
        print(data_set+' :')

        y_true = torch.tensor(y).cpu()
        y_pred = self.predict(arg1, arg2).cpu()

        torch.save(torch.tensor(confusion_matrix(y_true, y_pred)), data_set+'_confmat.pt')
        print(confusion_matrix(y_true, y_pred))

        torch.save(torch.tensor(f1_score(y_true, y_pred, average='macro')), data_set+'_f1macro.pt')
        print("f1 macro : ", f1_score(y_true, y_pred, average='macro'))

        torch.save(torch.tensor(precision_score(y_true, y_pred, average='macro')), data_set+'_precisionmacro.pt')
        print("precision macro : ", precision_score(y_true, y_pred, average='macro'))

        torch.save(torch.tensor(accuracy_score(y_true, y_pred)), data_set+'_accuracy.pt')
        print("exactitude : ", accuracy_score(y_true, y_pred))
        print()


# In[2]:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

MAX_LENGTH = 128

pdtb2 = []
reader = csv.DictReader(open('pdtb2.csv', 'r'))
for example in reader:
    pdtb2.append(example)


# In[3]:

# -------------------- Split ------------------------------

cb_split_set2sec = {"dev": [0, 1, 23, 24], "test": [21, 22], "train": list(range(2, 21))}
cb_split_sec2set = {}
for key, value in cb_split_set2sec.items():
    for section in value:
        cb_split_sec2set[section] = key

"""
print(cb_split_set2sec)
print(cb_split_sec2set)
"""

# In[31]:
"""
Chargement du fichier pdtb2
Arg1 contient les 1eres phrases, Arg2 les deuxiemes, y les goldclass
"""

Arg1, Arg2 = defaultdict(lambda: []), defaultdict(lambda: [])
X, y = defaultdict(lambda: []), defaultdict(lambda: [])
shuffle(pdtb2)
for example in pdtb2:
    if example['Relation'] == 'Implicit':
        Arg1[cb_split_sec2set[int(example['Section'])]].append(example['Arg1_RawText'])
        Arg2[cb_split_sec2set[int(example['Section'])]].append(example['Arg2_RawText'])
        y[cb_split_sec2set[int(example['Section'])]].append(example['ConnHeadSemClass1'].split('.')[0])


# In[32]:

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


# In[63]:
print(Arg1['train'][0])
# création du classifieur
discourse_relation_mlp = BertMLP(first_hidden_layer_size=50, second_hidden_layer_size=50, size_of_batch=100,
                                 dropout=0.3, loss=nn.NLLLoss(), device=device, num_classes=len(i2gold_class),
                                 Arg1train=Arg1['train'], Arg2train=Arg2['train'], ytrain=y['train'],
                                 Arg1dev=Arg1['dev'], Arg2dev=Arg2['dev'], ydev=y['dev'],
                                 i2goldclasses=i2gold_class)
discourse_relation_mlp = discourse_relation_mlp.to(device)

# quelques hyperparametres
learning_rate = 0.00003
l2_reg = 0.00005

# choix de l'optimizer (SGD, Adam, Autre ?)
optim = torch.optim.Adam(discourse_relation_mlp.parameters(), lr=learning_rate, weight_decay=l2_reg)

dev_losses, train_losses = discourse_relation_mlp.training_step(optimizer=optim, nb_epoch=1000, down_sampling=True)


# In[64]:
predict_train = discourse_relation_mlp.predict(Arg1['train'], Arg2['train'])
print(i2gold_class)


# In[65]:
discourse_relation_mlp.evaluation("train", Arg1["train"], Arg2["train"], y["train"])
discourse_relation_mlp.evaluation("dev", Arg1["dev"], Arg2["dev"], y["dev"])
discourse_relation_mlp.evaluation("test", Arg1["test"], Arg2["test"], y["test"])

# In[29]:
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
loss_fig.savefig('BertFineTunedModel.png')


# In[ ]:
# sauvegarde d'un modele
torch.save(discourse_relation_mlp, 'BertFineTuned_model.pth')

# chargement d'un modele
# discourse_relation_mlp = torch.load('fourth_model.pth')

# In[42]:
predict_NLI = discourse_relation_mlp.predict(Arg1['snli test'], Arg2['snli test'])
predict_revNLI = discourse_relation_mlp.predict(Arg2['snli test'], Arg1['snli test'])
# print(predict_NLI[:10])


# In[43]:
repartition = Counter([(nli_class, i2gold_class[int(disc_rel)])
                       for nli_class, disc_rel in zip(y_nli, predict_NLI.tolist())])
repartition_rev = Counter([(nli_class, i2gold_class[int(disc_rel)])
                           for nli_class, disc_rel in zip(y_nli, predict_revNLI.tolist())])
# print(repartition)


# In[40]:
# print(Counter(y_nli))

# In[54]:
i2nli = ['contradiction', 'entailment', 'neutral']


def save_plot(matrix, filename, index=i2gold_class, columns=['contradiction', 'entailment', 'neutral']):
    plt.figure()
    df_cm = DataFrame(matrix, index=index, columns=columns)
    ax = sn.heatmap(df_cm, cmap='Blues')
    heatmap = ax.get_figure()
    heatmap.savefig(filename, dpi=400)


mat = torch.tensor([[repartition[(nli_class, rel)] for rel in i2gold_class] for nli_class in i2nli])
print(mat)
save_plot(mat.T, 'AvantNormalisation.png')


mat1 = mat.T/torch.sum(mat, axis=1)
print(mat1)
save_plot(mat1, 'ApresNormalisationSNLI.png')

mat2 = mat/torch.sum(mat, axis=0)
print(mat2.T)
save_plot(mat1, 'ApresNormalisationPDTB.png')


mat = torch.tensor([[repartition_rev[(nli_class, rel)] for rel in i2gold_class] for nli_class in i2nli])
print(mat)
save_plot(mat.T, 'AvantNormalisation.png')


mat1 = mat.T/torch.sum(mat, axis=1)
print(mat1)
save_plot(mat1, 'ApresNormalisationSNLI.png')

mat2 = mat/torch.sum(mat, axis=0)
print(mat2.T)
save_plot(mat1, 'ApresNormalisationPDTB.png')