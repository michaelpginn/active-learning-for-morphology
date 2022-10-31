# Change this path
CONLL_SIGMORPHON_DATA_PATH="/Users/mpsilfve/src/conll2017/"

import click

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import log_softmax
from torch.optim import Adam, SGD

from random import random, seed, shuffle

from sklearn.metrics import accuracy_score

import torchtext
from torchtext.data import Field, ReversibleField
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator

# Padding symbol for batching, unknown symbol used for input characters, which were not attested in the training 
# set, and start & end of sequence symbols.
PAD="<pad>"
UNK="<unk>"
START="<start>"
END="<end>"

# Ensure reproducible results.
seed(0)
torch.manual_seed(0)
np.random.seed(0)

# Hyperparameters
EMBEDDING_DIM=50
RNN_HIDDEN_DIM=50
RNN_LAYERS=1

word_tok = lambda x: [c for c in x]
msd_tok = lambda x: ["FEAT=" + f for f in x.split(';')]

WORD = Field(sequential=True, tokenize=word_tok, lower=False,
             use_vocab=True,include_lengths=True, init_token=START,
             eos_token=END)
MSD = Field(sequential=True, tokenize=msd_tok, lower=False,use_vocab=True)

datafields = [("input", WORD), ("output", WORD), ("msd", MSD)]

def read_data(language,setting,batch_size=1):
    """Read shared task training, development and test sets for a particular language and
       return torchtext Iterators to the data. 
    """
    train, dev = TabularDataset.splits(
        path="%s/all/task1/" % CONLL_SIGMORPHON_DATA_PATH,
        train='%s-train-%s' % (language,setting),
        validation="%s-dev" % language,
        format='tsv',
        skip_header=True,
        fields=datafields)

    test = TabularDataset(
        path="%s/answers/task1/%s-uncovered-test" % (CONLL_SIGMORPHON_DATA_PATH,language),
        format='tsv',
        skip_header=True,
        fields=datafields)

    # Concatenate the lemma and MSD fields into a joint input field.
    for data in [train, dev, test]:
        for ex in data:
            ex.input = ex.input + ex.msd
    
    # Build vocabularies                                                        
    WORD.build_vocab(train)
    MSD.build_vocab(train)

    # Define train_iter, dev_iter and test_iter iterators over the training data, 
    # development data and test data, respectively.
    train_iter = BucketIterator(
        train,
        batch_size=batch_size,
        device="cpu",
        sort_key=lambda x: len(x.lemma),
        sort_within_batch=False,
        repeat=False)

    dev_iter, test_iter = Iterator.splits(
        (dev, test),
        batch_sizes=(batch_size,batch_size),
        device="cpu",
        sort=False,
        sort_within_batch=False,
        shuffle=False,
        repeat=False)
    
    return train_iter, dev_iter, test_iter

train_iter, dev_iter, test_iter = read_data(language="english",
                                            setting="medium",
                                            batch_size=1)
class Encoder(nn.Module):
    def __init__(self,alphabet):
        super(Encoder,self).__init__()
        self.c2i = alphabet.stoi
        self.i2c = alphabet.itos
        self.embedding = nn.Embedding(len(alphabet), EMBEDDING_DIM)
        self.fwd_rnn = nn.LSTM(EMBEDDING_DIM, RNN_HIDDEN_DIM, RNN_LAYERS)
        self.bwd_rnn = nn.LSTM(EMBEDDING_DIM, RNN_HIDDEN_DIM, RNN_LAYERS)
        self.fwd_prediction_layer = nn.Linear(RNN_HIDDEN_DIM,len(alphabet))
        self.bwd_prediction_layer = nn.Linear(RNN_HIDDEN_DIM,len(alphabet))

    def forward(self,ex):
        input, _ = ex.input
        embedded_input = self.embedding(input) 
        fwd_hidden_states, _ = self.fwd_rnn(embedded_input)
        bwd_hidden_states, _ = self.bwd_rnn(embedded_input.flip(dims=[0]))
        bwd_hidden_states = bwd_hidden_states.flip(dims=[0])
            
        fwd_distr = self.fwd_prediction_layer(fwd_hidden_states[:-1,:]).log_softmax(dim=2)
        bwd_distr = self.bwd_prediction_layer(bwd_hidden_states[1:,:]).log_softmax(dim=2)

        return torch.cat([fwd_hidden_states, bwd_hidden_states], dim=2), fwd_distr, bwd_distr

    def val_loss(self, data, r):
        loss_function = nn.NLLLoss(ignore_index=self.c2i[PAD],
                                   reduction='mean')
        loss = []
        for ex in data:
            input, _ = ex.input
            input = input.permute(1,0)
            _, fwd_enc_distr, bwd_enc_distr = self.forward(ex)
            fwd_enc_distr = fwd_enc_distr.permute(1,2,0)
            bwd_enc_distr = bwd_enc_distr.permute(1,2,0)
            fwd_loss = loss_function(fwd_enc_distr, input[:,1:])
            bwd_loss = loss_function(bwd_enc_distr, input[:,:-1])
            ex_loss = (fwd_loss + bwd_loss)/((input.size()[1] - 1)**r)
            loss.append(ex_loss.detach().numpy())
        return loss, np.average(loss)


    train_iter, dev_iter, test_iter = read_data(language="finnish",
                                            setting="high",
                                            batch_size=1)

def train_model(path, epochs):
    lm = Encoder(WORD.vocab)
    loss_function = nn.NLLLoss(ignore_index=lm.c2i[PAD],reduction='mean')
    optimizer = Adam(lm.parameters())
    gold_dev_words = [''.join(w.output) for w in dev_iter.dataset]

    for epoch in range(1, epochs+1):
        tot_loss = 0 
        print(f"EPOCH {epoch}")

        for batch in tqdm(list(train_iter)):
            lm.zero_grad()
            _, fwd_enc_distr, bwd_enc_distr = lm(batch)
            
            input, _ = batch.input
            input = input.permute(1,0)
            fwd_enc_distr = fwd_enc_distr.permute(1,2,0)
            bwd_enc_distr = bwd_enc_distr.permute(1,2,0)

            fwd_loss = loss_function(fwd_enc_distr, input[:,1:])
            bwd_loss = loss_function(bwd_enc_distr, input[:,:-1])
            loss = fwd_loss + bwd_loss
            tot_loss += loss.detach().numpy()
            loss.backward()
            optimizer.step()

        mean_train_loss = tot_loss/len(train_iter)        

        _, mean_val_loss = lm.val_loss(dev_iter, r=1.0)
        print(f"MEAN TRAIN LOSS: {mean_train_loss:.5f}")
        print(f"MEAN DEV LOSS: {mean_val_loss:.5f}")

    torch.save(lm, path)

def score_strings(model_path, data_path):
    read_data(language="english",
              setting="medium",
              batch_size=1)

    model = torch.load(model_path)

    test = TabularDataset(
        path=data_path,
        format='tsv',
        skip_header=True,
        fields=datafields)
    
    test_iter = Iterator.splits(
        (test,),
        batch_sizes=(1,),
        device="cpu",
        sort=False,
        sort_within_batch=False,
        shuffle=False,
        repeat=False)[0]

    # r is an adjustable hyperparameter actually
    losses, _ = model.val_loss(test_iter, r=1.0)
    for ex, loss in zip(test, losses):
        print(f"{''.join(ex.input)}\t{loss}")
    
@click.command()
@click.option("--mode", required=True)
@click.option("--path", required=True)
@click.option("--epochs", type=int, required=False)
@click.option("--data_path", required=False)
def main(mode, path, epochs, data_path):
    if mode == "train":
        train_model(path, epochs)
    elif mode == "test":
        score_strings(path, data_path)
    else:
        assert(0)
        

if __name__=="__main__":
    main()
