# Import libraries
import enum
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import ast
import numpy as np

vacuum_pid2brand = {'B00002N8CX': 'Eureka',
 'B00009R66F': 'Hoover',
 'B0000DF0RB': 'McCulloch',
 'B0028MB3HM': 'SharkNinja',
 'B003Y3AA2S': 'Bissell',
 'B003ZYPZ0I': 'Hoover',
 'B007L5I7DY': 'SharkNinja',
 'B00CK3WZ9Y': 'Ontel',
 'B00K316IB6': 'Bissell',
 'B00MFBJEU4': 'Rug Doctor',
 'B00SMLJPIC': 'Dyson',
 'B00SMLJQ7W': 'Dyson',
 'B00W7C7M3M': 'Eureka',
 'B010CZTYZY': 'Eureka',
 'B010T1QZAI': 'Black & Decker',
 'B0126C9A5K': 'Bissell',
 'B0168KHVZW': 'Neato Robotics',
 'B01BXBX6E6': 'Black & Decker',
 'B01DAI5BZ2': 'BLACK+DECKER',
 'B01DTYAZO4': 'Bissell'}

laptop_pid2brand = {'B00MMLV7VQ': 'Acer',
 'B00NSHLUBU': 'HP',
 'B011KFQASE': 'Asus',
 'B015806LSQ': 'Toshiba',
 'B015PYYDMQ': 'Dell',
 'B017QDHENY': 'HP',
 'B019G7VPTC': 'Acer',
 'B01CGGOZOM': 'HP',
 'B01CVOLVPA': 'Acer',
 'B01DBGVB7K': 'Asus'}
 
pid2brand = {"vacuum": vacuum_pid2brand, "laptop": laptop_pid2brand}

vacuum_brands = ['Eureka', 'Hoover', 'Dyson']
laptop_brands = ['Acer', 'HP', 'Dell']
brands = {"vacuum": vacuum_brands, "laptop": laptop_brands}

from collections import Counter, defaultdict
import re
empty_char_re = re.compile(r'\s+')
# create custom dataset class
class ReviewDataset(Dataset):
    bos_token = '<BOS>'
    eos_token = '<EOS>'
    unk_token = '<UNK>'
    def __init__(self, data_hparams, product_category='laptop', max_sents=10, max_len=50, vocab_size=20000, min_freq=10):
        reviews = pd.read_csv(data_hparams['file_path'])
        reviews['__brand']=reviews['asin'].map(pid2brand[product_category])
        # only keep selected brands
        reviews = reviews[reviews['__brand'].isin(brands[product_category])]
        reviews['__const'] = 1
        reviews['reviews_list'] = reviews['__filtered_review'].apply(ast.literal_eval)
        reviews['len'] = reviews['reviews_list'].apply(len)
        reviews = reviews[reviews['len'] > 0]
        raw_doc = reviews['reviews_list'].values
        tokenized_doc = []
        vocab_counter = Counter()
        for sents in raw_doc:
            tokenized_sents = []
            sents = sents[:max_sents]
            for sent in sents:
                sent = re.sub(empty_char_re, ' ', sent)
                tokens = sent.split(" ")[:max_len]
                vocab_counter.update(tokens)
                tokenized_sents.append([ReviewDataset.bos_token]+tokens+[ReviewDataset.eos_token])
            tokenized_doc.append(tokenized_sents)
        self.vocab = [ReviewDataset.unk_token, ReviewDataset.bos_token, ReviewDataset.eos_token] \
            + [x for x, freq in vocab_counter.most_common(vocab_size) if freq >= min_freq]
        self.vocab_dict = defaultdict(int, {x:idx for idx, x in enumerate(self.vocab)})
        self.text = []
        self.text_ids = []
        self.length = []
        for sents in tokenized_doc:
            filtered_sents = []
            ids_sents = []
            len_sents = []
            for sent in sents:
                sent_id = [self.vocab_dict[x] for x in sent]
                ids_sents.append(sent_id)
                filtered_sents.append([self.vocab[x] for x in sent_id])
                len_sents.append(len(sent_id))
            self.text.append(filtered_sents)
            self.text_ids.append(ids_sents)
            self.length.append(len_sents)
        
        self.bos_token_id = self.vocab_dict[ReviewDataset.bos_token]
        self.eos_token_id = self.vocab_dict[ReviewDataset.eos_token]

        self.labels = reviews['overall'].values
        brand_dummies = pd.get_dummies(reviews['__brand'], prefix='brand')
        pid_dummies = pd.get_dummies(reviews['asin'], prefix='pid')
        self.dummies = pd.concat((brand_dummies, pid_dummies), axis=1).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.text_ids[idx]
        text = self.text[idx]
        length = self.length[idx]
        sample = {"text": text, "text_ids": data, "label": label, "dummy": self.dummies[idx], 'length': length}
        return sample
    
    def map_ids_to_tokens_py(self, sample_ids):
        sample_tokens = []
        for sent in sample_ids:
            token_sent = []
            try:
                eos_idx = sent.index(self.eos_token_id)
                token_sent = [self.vocab[id] for id in sent[:eos_idx+1]]
            except ValueError:
                trailing_zeros = True
                for id in reversed(sent):
                    if trailing_zeros:
                        if id == 0:
                            continue
                        else:
                            trailing_zeros = False
                    token_sent.insert(0, self.vocab[id])
            sample_tokens.append(token_sent)
        return sample_tokens

# collate_fn
def collate_batch(batch):
    max_len = max(sum([x['length'] for x in batch], []))
    text_ids_list = []
    text_list = []
    labels = []
    dummies = []
    lens = []
    for item in batch:
        for tid, t in zip (item['text_ids'], item['text']):
            text_ids_list.append(tid+[0]*(max_len - len(tid)))
            text_list.append(t+[ReviewDataset.unk_token]*(max_len - len(t)))
        labels.append(item['label'])
        dummies.append(item['dummy'])
        lens.append(item['length'])

    return {"text": text_list, "text_ids": torch.tensor(text_ids_list), "label": torch.tensor(labels), 
    "dummy": torch.tensor(np.vstack(dummies)), "length": torch.tensor(sum(lens, []))}

# # create DataLoader object of DataSet object
# bat_size = 2
# RD = ReviewDataset("./Language_modeling/Text_gen_PTB/simple-examples/data/top3_laptop_reviews_pyABSA_filtered_lemma_lg.csv")
# DL_DS = DataLoader(RD, batch_size=bat_size, shuffle=False, collate_fn=collate_batch)

# # loop through each batch in the DataLoader object
# for (idx, batch) in enumerate(DL_DS):

#     # Print the 'text' data of the batch
#     print(idx, 'Text data: ', batch, '\n')

#     break