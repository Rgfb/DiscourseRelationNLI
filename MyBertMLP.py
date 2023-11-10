# --------------------- Installations et Imports -------------------------
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle
from collections import Counter, defaultdict
from transformers import AutoModel, AutoTokenizer


# ----------------------- Le Modèle ---------------------------------

"""
un MLP qui prend en entrée une phrase tokenisée et renvoie la liste des probas pour certaines classes du PDTB2
(à voir quelles classes nous intéressent le plus ..)

"""


class BertMLP(nn.Module):

    def __init__(self, first_hidden_layer_size, second_hidden_layer_size, size_of_batch, dropout, device, num_classes,
                 Arg1train, Arg2train, ytrain, Arg1dev, Arg2dev, ydev, size_of_input=768, num_tokens=128,
                 loss=nn.NLLLoss(), model_name="bert-base-uncased", reg=1):

        super(BertMLP, self).__init__()

        self.loss = loss
        self.Arg1train, self.Arg2train, self.ytrain = Arg1train, Arg2train, ytrain
        self.Arg1dev, self.Arg2dev, self.ydev = Arg1dev, Arg2dev, ydev

        self.num_classes = num_classes
        self.num_tokens = num_tokens
        self.size_of_batch = size_of_batch

        self.w1 = nn.Linear(size_of_input, first_hidden_layer_size).to(device)
        self.w2 = nn.Linear(first_hidden_layer_size, second_hidden_layer_size)
        self.w3 = nn.Linear(second_hidden_layer_size, num_classes).to(device)

        self.dropout = nn.Dropout(dropout).to(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name).to(device)

        self.reg = reg
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

    def sampling(self, examples, size_of_samples):
        # creation du sample sur lequel on va s'entrainer
        arg1_sample = []
        arg2_sample = []
        y_sample = []

        for gold_class in range(self.num_classes):
            # shuffle pour ne pas prendre les mêmes exemples à chaque fois
            shuffle(examples[gold_class])

            for vector in examples[gold_class][:size_of_samples]:
                arg1_sample.append(vector[0])
                arg2_sample.append(vector[1])
                y_sample.append(gold_class)

        return arg1_sample, arg2_sample, y_sample

    def training_step(self, optimizer, nb_epoch=2, patience=2,
                      down_sampling=True, fixed_sampling=False, size_of_samples=2000):

        """
        l'entrainement du MLP

        reg : regularite du calcul de la loss (on calcule la loss toutes les reg epoques)
        down_sampling : booleen pour savoir si on fait du down sampling
        size_of_samples : taille des samples lorsqu'on fait du down sampling
        """

        # les listes qui contiendront les valeurs de la loss sur le dev et le train pour chaque époque
        dev_losses = []
        train_losses = []

        # est-ce qu'on veut faire du down sampling ou non
        if down_sampling:
            examples = defaultdict(lambda: [])
            for arg1, arg2, label in zip(self.Arg1train, self.Arg2train, self.ytrain):
                examples[label].append((arg1, arg2))

            if fixed_sampling:
                # creation du sample sur lequel on va s'entrainer
                arg1_sample, arg2_sample, y_sample = self.sampling(examples, size_of_samples)

        else:
            arg1_sample, arg2_sample, y_sample = self.Arg1train, self.Arg2train, self.ytrain

        for epoch in range(nb_epoch):

            # est-ce qu'on veut faire du downsampling fixe ou non
            if down_sampling and not fixed_sampling:
                # creation du sample sur lequel on va s'entrainer
                arg1_sample, arg2_sample, y_sample = self.sampling(examples, size_of_samples)

            # melange du sample pour ne pas toujours s'entrainer sur les labels dans le même ordre
            sample = list(zip(arg1_sample, arg2_sample, y_sample))
            shuffle(sample)
            arg1_sample, arg2_sample, y_sample = zip(*sample)

            # mode "train", on précise quand on s'entraine et quand on évalue le modèle pour le dropout
            self.train()

            # un indice i pour parcourir tout le sample
            i = 0
            while i < len(y_sample):
                # arg1 (resp arg2) : les tokens des premieres (resp deuxiemes) phrases tokenisees
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

            # calcul et print régulier de la loss sur le train et le dev
            if epoch % self.reg == 0:
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

        y_true = torch.tensor(y).cpu()
        y_pred = self.predict(arg1, arg2).cpu()

        print(data_set+' :')
        print(confusion_matrix(y_true, y_pred))
        print("f1 macro : ", f1_score(y_true, y_pred, average='macro'))
        print("precision macro : ", precision_score(y_true, y_pred, average='macro'))
        print("exactitude : ", accuracy_score(y_true, y_pred))
        print()
