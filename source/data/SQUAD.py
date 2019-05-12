import torch
from torch.utils.data import Dataset
import numpy as np
import random
import json


class SQuADDataset(Dataset):
    def __init__(self, npz_file, num_steps, batch_size, word2ind_file):
        super().__init__()
        data = np.load(npz_file)
        with open(word2ind_file) as f:
            decoder = json.load(f)
        self.dictionary = dict((item, key) for key, item in decoder.items())
        self.context_idxs = torch.from_numpy(data["context_idxs"]).long()
        self.context_char_idxs = torch.from_numpy(data["context_char_idxs"]).long()
        self.ques_idxs = torch.from_numpy(data["ques_idxs"]).long()
        self.ques_char_idxs = torch.from_numpy(data["ques_char_idxs"]).long()
        self.y1s = torch.from_numpy(data["y1s"]).long()
        self.y2s = torch.from_numpy(data["y2s"]).long()
        self.ids = torch.from_numpy(data["ids"]).long()
        num = len(self.ids)
        self.batch_size = batch_size
        self.num_steps = num_steps if num_steps >= 0 else num // batch_size
        num_items = num_steps * batch_size
        idxs = list(range(num))
        self.idx_map = []
        i, j = 0, num

        while j <= num_items:
            random.shuffle(idxs)
            self.idx_map += idxs.copy()
            i = j
            j += num
        random.shuffle(idxs)
        self.idx_map += idxs[:num_items - i]

    def __len__(self):
        return self.num_steps

    def __getitem__(self, item):
        idxs = torch.LongTensor(self.idx_map[item:item + self.batch_size])
        res = (self.context_idxs[idxs],
               self.context_char_idxs[idxs],
               self.ques_idxs[idxs],
               self.ques_char_idxs[idxs],
               self.y1s[idxs],
               self.y2s[idxs], self.ids[idxs])
        return res
    
    def decode(self, tensor):
        tensor = tensor.tolist()[0]
        result = []
        for key in tensor:
            if self.dictionary[key] == '--NULL--':
                break
            result.append(self.dictionary[key])
        return ' '.join(result)
