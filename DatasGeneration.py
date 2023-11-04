# --------------------- Installations et Imports -------------------------

import csv
from collections import defaultdict
import pandas as pd


# ---------------- Lecture et Partitionnement des fichiers ----------------------
class PDTBReader:

    def __init__(self):
        """
        Arg1 contient les 1eres phrases, Arg2 les deuxiemes,
        y les goldclass (ie les relations de discours et les connecteurs qui nous interessent pour le PDTB)
        """
        self.Arg1, self.Arg2 = defaultdict(lambda: []), defaultdict(lambda: [])
        self.rel, self.conn = defaultdict(lambda: []), defaultdict(lambda: [])

    def read(self, split='CB', relation='Implicit', conn_filter=[], class_filter=[]):
        pdtb = []
        # Chargement du fichier pdtb2
        reader = csv.DictReader(open('datas/pdtb2.csv/pdtb2.csv', 'r'))
        for example in reader:
            pdtb.append(example)

        """
        split : le partitionnement des données (CB pour celui de Chloé Braud,
                                                Lin pour celui de Lin et al,
                                                Ji pour celui de Ji et al)

        on définit :
        - split_sec2set : a une section (0 a 24) associe son set (train, dev ou test)
        - split_set2sec : a un set (train, dev ou test) associe la liste des sections qui lui sont associées
        """

        if split == 'CB':
            split_set2sec = {"dev": [0, 1, 23, 24], "test": [21, 22], "train": list(range(2, 21))}
        if split == 'Lin':
            split_set2sec = {"dev": [22], "test": [23], "train": list(range(2, 22))}
        if split == 'Ji':
            split_set2sec = {"dev": [0, 1], "test": [21, 22], "train": list(range(2, 21))}
        split_sec2set = {}
        for key, value in split_set2sec.items():
            for section in value:
                split_sec2set[section] = key

        for example in pdtb:
            section = int(example['Section'])
            if example['Relation'] == relation and section in split_sec2set:

                sem_class = example['ConnHeadSemClass1'].split('.')[0]
                # Conn1 dans le cas implicite, ConnHead dans le cas explicite
                connective = example['Conn1'] if relation == 'Implicit' else example['ConnHead']

                if connective in conn_filter and sem_class not in class_filter:
                    self.Arg1[relation + '_' + split_sec2set[section]].append(example['Arg1_RawText'])
                    self.Arg2[relation + '_' + split_sec2set[section]].append(example['Arg2_RawText'])
                    self.rel[relation + '_' + split_sec2set[section]].append(sem_class)

                    self.conn[relation + '_' + split_sec2set[section]].append(connective)


class SNLIReader:

    def __init__(self):
        """
        Arg1 contient les 1eres phrases, Arg2 les deuxiemes,
        y les goldclass (ie les relations de discours qui nous interessent pour le PDTB,
        et les classes NLI pour le SNLI)
        """
        self.Arg1, self.Arg2, self.y = defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: [])

    def read(self, part='dev'):
        """
        Lecture de la partie (train, test, ou dev) qui nous intéresse
        les colonnes qui nous intéressent :
            les 2 phrases (dans l'ordre)
            la goldclass
        """
        file = pd.read_csv("datas/snli_1.0/snli_1.0/snli_1.0_" + part + ".txt", sep="\t")
        file = file[['gold_label', 'sentence1', 'sentence2']]
        for gold, sent1, sent2 in zip(file['gold_label'], file['sentence1'], file['sentence2']):
            if gold != '-':
                if isinstance(sent2, float):
                    print(sent1, '\n', sent2, '\n', gold, '\n')
                else:
                    self.Arg1[part].append(sent1)
                    self.Arg2[part].append(sent2)
                    self.y[part].append(gold)
