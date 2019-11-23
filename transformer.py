import itertools, os, time , datetime
import numpy as np
import spacy
import torch
import torch.nn as nn
from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe
use_gpu = torch.cuda.is_available()

def preprocess(vocab_size=0, batchsize=16, max_sent_len=20):
    '''Loads data from text files into iterators'''

    # Load text tokenizers
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize(text, lang='en'):
        if lang is 'de':
            return [tok.text for tok in spacy_de.tokenizer(text)]
        elif lang is 'en':
            return [tok.text for tok in spacy_en.tokenizer(text)]
        else:
            raise Exception('Invalid language')

    # Add beginning-of-sentence and end-of-sentence tokens 
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    DE = data.Field(tokenize=lambda x: tokenize(x, 'de'))
    EN = data.Field(tokenize=tokenize, init_token=BOS_WORD, eos_token=EOS_WORD)

    # Create sentence pair dataset with max length 20
    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), filter_pred = lambda x: max(len(vars(x)['src']), len(vars(x)['trg'])) <= max_sent_len)

    # Build vocabulary and convert text to indices
    # Convert words that appear fewer than 5 times to <unk>
    if vocab_size > 0:
        DE.build_vocab(train.src, min_freq=5, max_size=vocab_size)
        EN.build_vocab(train.trg, min_freq=5, max_size=vocab_size)
    else:
        DE.build_vocab(train.src, min_freq=5)
        EN.build_vocab(train.trg, min_freq=5)

    # Create iterators to process text in batches of approx. the same length
    train_iter = data.BucketIterator(train, batch_size=batchsize, device=-1, repeat=False, sort_key=lambda x: len(x.src))
    val_iter = data.BucketIterator(val, batch_size=1, device=-1, repeat=False, sort_key=lambda x: len(x.src))
    
    return DE, EN, train_iter, val_iter

# Test
timer = time.time()
SRC, TGT, train_iter, val_iter = preprocess()

print('''This is a test of our preprocessing function. It took {:.1f} seconds to load the data. 
Our German vocab has size {} and our English vocab has size {}.
Our training data has {} batches, each with {} sentences, and our validation data has {} batches.'''.format(
time.time() - timer, len(SRC.vocab), len(TGT.vocab), len(train_iter), train_iter.batch_size, len(val_iter)))


def load_embeddings(SRC, TGT, np_src_file, np_tgt_file):
    emb_tr_src = torch.from_numpy(np.load(np_src_file))
    emb_tr_tgt = torch.from_numpy(np.load(np_tgt_file))
    return emb_tr_src, emb_tr_tgt


class EncoderLSTM(nn.Module):
    def __init__(self, embedding, h_dim, num_layers, dropout_p=0.0, bidirectional=True):
        super(EncoderLSTM, self).__init__()
        self.vocab_size, self.embedding_size = embedding.size()
        self.num_layers, self.h_dim, self.dropout_p, self.bidirectional = num_layers, h_dim, dropout_p, bidirectional 

        # Create embedding and LSTM
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(embedding)
        self.lstm = nn.LSTM(self.embedding_size, self.h_dim, self.num_layers, dropout=self.dropout_p, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        '''Embed text, get initial LSTM hidden state, and encode with LSTM'''
        x = self.dropout(self.embedding(x)) # embedding
        h0 = self.init_hidden(x.size(1)) # initial state of LSTM
        memory_bank, h = self.lstm(x, h0) # encoding
        return memory_bank, h

    def init_hidden(self, batch_size):
        '''Create initial hidden state of zeros: 2-tuple of num_layers x batch size x hidden dim'''
        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
        init = torch.zeros(num_layers, batch_size, self.h_dim)
        init = init.cuda() if use_gpu else init
        h0 = (init, init.clone())

class Attention(nn.Module):
    def __init__(self, pad_token=1, bidirectional=True, h_dim=300):
        super(Attention, self).__init__()
        self.bidirectional, self.h_dim, self.pad_token = bidirectional, h_dim, pad_token
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_e, out_e, out_d):
        '''Produces context with attention distribution'''

        # Deal with bidirectional encoder, move batches first
        if self.bidirectional: # sum hidden states for both directions
            out_e = out_e.contiguous().view(out_e.size(0), out_e.size(1), 2, -1).sum(2).view(out_e.size(0), out_e.size(1), -1)
            
        # Move batches first
        out_e = out_e.transpose(0,1) # b x sl x hd
        out_d = out_d.transpose(0,1) # b x tl x hd

        # Dot product attention, softmax, and reshape
        attn = out_e.bmm(out_d.transpose(1,2)) # (b x sl x hd) (b x hd x tl) --> (b x sl x tl)
        attn = self.softmax(attn).transpose(1,2) # --> b x tl x sl

        # Get attention distribution
        context = attn.bmm(out_e) # --> b x tl x hd
        context = context.transpose(0,1) # --> tl x b x hd
        return context

class DecoderLSTM(nn.Module):
    def __init__(self, embedding, h_dim, num_layers, dropout_p=0.0):
        super(DecoderLSTM, self).__init__()
        self.vocab_size, self.embedding_size = embedding.size()
        self.num_layers, self.h_dim, self.dropout_p = num_layers, h_dim, dropout_p
        
        # Create embedding and LSTM
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(embedding) 
        self.lstm = nn.LSTM(self.embedding_size, self.h_dim, self.num_layers, dropout=self.dropout_p)
        self.dropout = nn.Dropout(self.dropout_p)
    
    def forward(self, x, h0):
        '''Embed text and pass through LSTM'''
        x = self.embedding(x)
        x = self.dropout(x)
        out, h = self.lstm(x, h0)
        return out, h

class Seq2seq(nn.Module):
    def __init__(self, embedding_src, embedding_tgt, h_dim, num_layers, dropout_p, bi, tokens_bos_eos_pad_unk=[0,1,2,3]):
        super(Seq2seq, self).__init__()
        # Store hyperparameters
        self.h_dim = h_dim
        self.vocab_size_tgt, self.emb_dim_tgt = embedding_tgt.size()
        self.bos_token, self.eos_token, self.pad_token, self.unk_token = tokens_bos_eos_pad_unk

        # Create encoder, decoder, attention
        self.encoder = EncoderLSTM(embedding_src, h_dim, num_layers, dropout_p=dropout_p, bidirectional=bi)
        self.decoder = DecoderLSTM(embedding_tgt, h_dim, num_layers * 2 if bi else num_layers, dropout_p=dropout_p)
        self.attention = Attention(pad_token=self.pad_token, bidirectional=bi, h_dim=self.h_dim)

        # Create linear layers to combine context and hidden state
        self.linear1 = nn.Linear(2 * self.h_dim, self.emb_dim_tgt)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(self.emb_dim_tgt, self.vocab_size_tgt)
        
        # Share weights between decoder embedding and output 
        if self.decoder.embedding.weight.size() == self.linear2.weight.size():
            self.linear2.weight = self.decoder.embedding.weight

    def forward(self, src, tgt):
        if use_gpu: src = src.cuda()
        
        # Encode
        out_e, final_e = self.encoder(src)
        
        # Decode
        out_d, final_d = self.decoder(tgt, final_e)
        
        # Attend
        context = self.attention(src, out_e, out_d)
        out_cat = torch.cat((out_d, context), dim=2) 
        
        # Predict (returns probabilities)
        x = self.linear1(out_cat)
        x = self.dropout(self.tanh(x))
        x = self.linear2(x)
        return x

    def predict(self, src, beam_size=1): 
        '''Predict top 1 sentence using beam search. Note that beam_size=1 is greedy search.'''
        beam_outputs = self.beam_search(src, beam_size, max_len=30) # returns top beam_size options (as list of tuples)
        top1 = beam_outputs[0][1] # a list of word indices (as ints)
        return top1

    def beam_search(self, src, beam_size, max_len, remove_tokens=[]):
        '''Returns top beam_size sentences using beam search. Works only when src has batch size 1.'''
        if use_gpu: src = src.cuda()
        
        # Encode
        outputs_e, states = self.encoder(src) # batch size = 1
        
        # Start with '<s>'
        init_lprob = -1e10
        init_sent = [self.bos_token]
        best_options = [(init_lprob, init_sent, states)] # beam
        
        # Beam search
        k = beam_size # store best k options
        for length in range(max_len): # maximum target length
            options = [] # candidates 
            for lprob, sentence, current_state in best_options:
                # Prepare last word
                last_word = sentence[-1]
                if last_word != self.eos_token:
                    last_word_input = torch.LongTensor([last_word]).view(1,1)
                    if use_gpu: last_word_input = last_word_input.cuda()
                    # Decode
                    outputs_d, new_state = self.decoder(last_word_input, current_state)
                    # Attend
                    context = self.attention(src, outputs_e, outputs_d)
                    out_cat = torch.cat((outputs_d, context), dim=2)
                    x = self.linear1(out_cat)
                    x = self.dropout(self.tanh(x))
                    x = self.linear2(x)
                    x = x.squeeze().data.clone()
                    # Block predictions of tokens in remove_tokens
                    for t in remove_tokens: x[t] = -10e10
                    lprobs = torch.log(x.exp() / x.exp().sum()) # log softmax
                    # Add top k candidates to options list for next word
                    for index in torch.topk(lprobs, k)[1]: 
                        option = (float(lprobs[index]) + lprob, sentence + [index], new_state) 
                        options.append(option)
                else: # keep sentences ending in '</s>' as candidates
                    options.append((lprob, sentence, current_state))
            options.sort(key = lambda x: x[0], reverse=True) # sort by lprob
            best_options = options[:k] # place top candidates in beam
        best_options.sort(key = lambda x: x[0], reverse=True)
        return best_options

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_iter, val_iter, model, criterion, optimizer, num_epochs):  
    for epoch in range(num_epochs):
      
        # Validate model
        with torch.no_grad():
          val_loss = validate(val_iter, model, criterion) 
          print('Validating Epoch [{e}/{num_e}]\t Average loss: {l:.3f}\t Perplexity: {p:.3f}'.format(
            e=epoch, num_e=num_epochs, l=val_loss, p=torch.FloatTensor([val_loss]).exp().item()))

        # Train model
        model.train()
        losses = AverageMeter()
        for i, batch in enumerate(train_iter): 
            src = batch.src.cuda() if use_gpu else batch.src
            tgt = batch.trg.cuda() if use_gpu else batch.trg
            
            # Forward, backprop, optimizer
            model.zero_grad()
            scores = model(src, tgt)

            # Remove <s> from target and </s> from scores (output)
            scores = scores[:-1]
            tgt = tgt[1:]           

            # Reshape for loss function
            scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
            tgt = tgt.view(scores.size(0))

            # Pass through loss function
            loss = criterion(scores, tgt) 
            loss.backward()
            losses.update(loss.item())

            # Clip gradient norms and step optimizer
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Log within epoch
            if i % 1000 == 10:
                print('''Epoch [{e}/{num_e}]\t Batch [{b}/{num_b}]\t Loss: {l:.3f}'''.format(e=epoch+1, num_e=num_epochs, b=i, num_b=len(train_iter), l=losses.avg))

        # Log after each epoch
        print('''Epoch [{e}/{num_e}] complete. Loss: {l:.3f}'''.format(e=epoch+1, num_e=num_epochs, l=losses.avg))

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.do = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, sent len, hid dim]
        
        x = self.do(torch.relu(self.fc_1(x)))
        
        #x = [batch size, sent len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, sent len, hid dim]
        
        return x

def validate(val_iter, model, criterion):
    '''Calculate losses by teacher forcing on the validation set'''
    model.eval()
    losses = AverageMeter()
    for i, batch in enumerate(val_iter):
        src = batch.src.cuda() if use_gpu else batch.src
        tgt = batch.trg.cuda() if use_gpu else batch.trg
        
        # Forward 
        scores = model(src, tgt)
        scores = scores[:-1]
        tgt = tgt[1:]           
        
        # Reshape for loss function
        scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
        tgt = tgt.view(scores.size(0))
        num_words = (tgt != 0).float().sum()
        
        # Calculate loss
        loss = criterion(scores, tgt) 
        losses.update(loss.item())
    
    return losses.avg

def predict_from_text(model, input_sentence, SRC, TGT):
    sent_german = input_sentence.split(' ') # sentence --> list of words
    sent_indices = [SRC.vocab.stoi[word] if word in SRC.vocab.stoi else SRC.vocab.stoi['<unk>'] for word in sent_german]
    sent = torch.LongTensor([sent_indices])
    if use_gpu: sent = sent.cuda()
    sent = sent.view(-1,1) # reshape to sl x bs
    print('German: ' + ' '.join([SRC.vocab.itos[index] for index in sent_indices]))
    # Predict five sentences with beam search 
    pred = model.predict(sent, beam_size=5) # returns list of 5 lists of word indices
    out = ' '.join([TGT.vocab.itos[index] for index in pred[1:-1]])
    print('English: ' + out)

embedding_src, embedding_tgt = load_embeddings(SRC, TGT, 'emb-13353-de.npy', 'emb-11560-en.npy')

tokens = [TGT.vocab.stoi[x] for x in ['<s>', '</s>', '<pad>', '<unk>']]
model = Seq2seq(embedding_src, embedding_tgt, 300, 2, 0.3, True, tokens_bos_eos_pad_unk=tokens)
model = model.cuda() if use_gpu else model

weight = torch.ones(len(TGT.vocab))
weight[TGT.vocab.stoi['<pad>']] = 0
weight = weight.cuda() if use_gpu else weight

criterion = nn.CrossEntropyLoss(weight=weight)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

train(train_iter, val_iter, model, criterion, optimizer, 50)

with torch.no_grad():
    val_loss = validate(val_iter, model, criterion) 
    print('Average loss: {l:.3f}\t Perplexity: {p:.3f}'.format(l=val_loss, p=torch.FloatTensor([val_loss]).exp().item()))

input = "Ich kenne nur Berge, ich bleibe in den Bergen und ich liebe die Berge ."
predict_from_text(model, input, SRC, TGT)

input = "Ihre Bergung erwies sich als komplizierter als gedacht ." 
predict_from_text(model, input, SRC, TGT)