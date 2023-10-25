# --------------------- Installations et Imports -------------------------
import csv
from collections import defaultdict


class PDTBRreader:

    def __init__(self):
        self.pdtb = []
        # Chargement du fichier pdtb2
        reader = csv.DictReader(open('datas/pdtb2.csv/pdtb2.csv', 'r'))
        for example in reader:
            self.pdtb.append(example)

    def split(self, split='CB'):

        """
        split : le partitionnement des données (CB pour celui de Chloé Braud,
                                                Lin pour celui de Lin et al,
                                                Ji pour celui de Ji et al)

        on définit :
        - split_sec2set : a une section (0 a 24) associe son set (train, dev ou test)
        - split_set2sec : a un set (train, dev ou test) associe la liste des sections qui lui sont associées

        """
        if split == 'CB':
            self.split_set2sec = {"dev": [0, 1, 23, 24], "test": [21, 22], "train": list(range(2, 21))}
        if split == 'Lin':
            self.split_set2sec = {"dev": [22], "test": [23], "train": list(range(2, 22))}
        if split == 'Ji':
            self.split_set2sec = {"dev": [0, 1], "test": [21, 22], "train": list(range(2, 21))}
        self.split_sec2set = {}
        for key, value in self.split_set2sec.items():
            for section in value:
                self.split_sec2set[section] = key

        """
        Arg1 contient les 1eres phrases, Arg2 les deuxiemes, 
        y les goldclass (ie les relations de discours qui nous interessent)
        """
        Arg1, Arg2, y = defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: [])
        for example in self.pdtb:
            section = int(example['Section'])
            if example['Relation'] == 'Implicit' and section in self.split_sec2set:
                gold_class = example['ConnHeadSemClass1'].split('.')[0]
                Arg1[self.split_sec2set[section]].append(example['Arg1_RawText'])
                Arg2[self.split_sec2set[section]].append(example['Arg2_RawText'])
                y[self.split_sec2set[section]].append(gold_class)
        return Arg1, Arg2, y



