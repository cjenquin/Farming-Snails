#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# packages and GPU initialization

import numpy as np
import pandas as pd
import re
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random as rand

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.manual_seed(103)
torch.cuda.manual_seed(103)
np.random.seed(103)

deviceCount = torch.cuda.device_count()
print(deviceCount)

cuda0 = None
if deviceCount > 0:
    print(torch.cuda.get_device_name(0))
    cuda0 = torch.device('cuda:0')


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# In[ ]:


#-----------------------------------------------------------------------------------------------------------------------------#
#--DATA FORMATTING------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#


# In[ ]:


df = pd.read_csv("philosophy_data.csv")
df['y_expected'] = df['tokenized_txt']      # ensures that the new column can store a list datatype

drop_list = []
for i in range(df.shape[0]):
    author = df.at[i,'author']
    if author=='Aristotle':
        df.at[i,'y_expected'] = torch.tensor([1,0,0,0,0])
    elif author=='Plato':
        df.at[i,'y_expected'] = torch.tensor([0,1,0,0,0])
    elif author=='Hegel':
        df.at[i,'y_expected'] = torch.tensor([0,0,1,0,0])
    elif author=='Foucault':
        df.at[i,'y_expected'] = torch.tensor([0,0,0,1,0])
    elif author=='Heidegger':
        df.at[i,'y_expected'] = torch.tensor([0,0,0,0,1])
    else:
        drop_list.append(i) 
        
df = df.drop(drop_list)


# In[ ]:


# clean and tokenize each text entry

def clean_text(text):
    
    # lower case characters only
    text = text.lower() 
    
    # remove urls
    text = re.sub('http\S+', ' ', text)
    
    # only alphabets, spaces and apostrophes 
    text = re.sub("[^a-z' ]+", ' ', text)
    
    # remove all apostrophes which are not used in word contractions
    text = ' ' + text + ' '
    text = re.sub("[^a-z]'|'[^a-z]", ' ', text)
    
    return text.split()


df['tokenized_txt'] = df['sentence_str'].apply(lambda x: clean_text(x))


# In[ ]:


# expand most common contractions in text entries

contractions  = { "i'm" : "i am", "it's" : "it is", "don't" : "do not", "can't" : "cannot", 
                  "you're" : "you are", "that's" : "that is", "we're" : "we are", "i've" : "i have", 
                  "he's" : "he is", "there's" : "there is", "i'll" : "i will", "i'd" : "i would", 
                  "doesn't" : "does not", "what's" : "what is", "didn't" : "did not", 
                  "wasn't" : "was not", "hasn't" : "has not", "they're" : "they are", 
                  "let's" : "let us", "she's" : "she is", "isn't" : "is not", "ain't" : "not", 
                  "aren't" : "are not", "haven't" : "have not", "you'll" : "you will", 
                  "we've" : "we have", "you've" : "you have", "y'all" : "you all", 
                  "weren't" : "were not", "couldn't" : "could not", "would've" : "would have", 
                  "they've" : "they have", "they'll" : "they will", "you'd" : "you would", 
                  "they'd" : "they would", "it'll" : "it will", "where's" : "where is", 
                  "we'll" : "we will", "we'd" : "we would", "he'll" : "he will", "shouldn't" : "should not", 
                  "wouldn't" : "would not", "won't" : "will not" }


def expand_contractions(words):
    
    for i in range(len(words)):
        if words[i] in contractions:
            words[i] = contractions[words[i]]
            
    return (' '.join(words)).split()


# precautionary cleaning for any remaing apostrophes
def remove_apostrophes(words):
    words = ' '.join(words)
    words = re.sub("'", '', words)
    return words.split()


df['tokenized_txt'] = df['tokenized_txt'].apply(lambda words: expand_contractions(words))
df['tokenized_txt'] = df['tokenized_txt'].apply(lambda words: remove_apostrophes(words))


# In[ ]:


# remove entries still longer than 300 words

entry_len = 300

drop_list = []
for i in range(df.shape[0]):
    if i in df and len(df.at[i,'tokenized_txt']) >= entry_len:
        drop_list.append(i)
        
df = df.drop(drop_list).reset_index(drop = True)


# In[ ]:


# remove entries still longer than 100 words

entry_len = 100

drop_list = []
for i in range(df.shape[0]):
    if len(df.at[i,'tokenized_txt']) >= entry_len:
        drop_list.append(i)
        
df = df.drop(drop_list).reset_index(drop = True)


# In[ ]:


count = 0
for i in df['tokenized_txt']:
    if len(i) >100:
        count += 1
        
print(count)


# In[ ]:


text_embed_dim = 100                                          # changes the embedding dimension used
glove_file = 'glove.6B.{}d.txt'.format(text_embed_dim)    # defines the GloVe file path -- change if using a new encoding dataset

embed_dict = {}
with open(glove_file, encoding = "utf8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        
        # removed dtype = np.float64 from the call. 
        coefs = np.fromstring(coefs, dtype = np.float32, sep=" ")
        embed_dict[word] = coefs
        
print("Found %s word vectors." % len(embed_dict))


# In[ ]:


#-----------------------------------------------------------------------------------------------------------------------------#
#--NON-RANDOM DATA SELECTION--------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#


# In[ ]:


# select the first 10,000 entries for each of our philosophers
num_entries = 10000

author_names = list(set(df['author']))
df_list = [df[df['author'] == author][:num_entries].reset_index(drop=True) for author in author_names]
df = pd.concat(df_list).reset_index(drop = True)


# In[ ]:


len(df['author'])


# In[ ]:


entry_len = 100

# replaces strings in an input sentence with word vectors
def embeddings(sent):
    
    # embedding includes an internal padding step
    padding_vec = np.zeros(text_embed_dim)
    out_list = [padding_vec] * entry_len
    
    # no padding step
    #out_list = []
    
    for j in range (0, len(sent)):
        if sent[j] in embed_dict:
            out_list[j] = embed_dict[sent[j]]
            
            
    return(torch.tensor([[out_list]]))


# In[ ]:


# this is just list storage of lables for the same data 
identity_list = [df['y_expected'].values[i] for i in range (5*num_entries)]


# In[ ]:


# this step applies embedding and padding to all tokens
tokens_list = [embeddings(df['tokenized_txt'].values[i]) for i in range (5*num_entries)]


# In[ ]:


#-----------------------------------------------------------------------------------------------------------------------------#
#--DATA LIST POPULATION-------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#


# In[ ]:


# Now we make our special dataset where authors are equally represented (in number of sentences)
dataset_len = len(df['tokenized_txt'])

num_authors = 5

test_len = int(num_entries/num_authors)
valid_len = int(num_entries/num_authors)
train_len = int((3*num_entries/num_authors))



test_list = [[],[]]
valid_list = [[],[]]
train_list = [[],[]]


# In[ ]:


# populate list of test sentences
for i in range (0, num_authors):
    for j in range (0, test_len):
        test_list[0].append(tokens_list[i * num_entries + j])
        test_list[1].append(identity_list[i * num_entries + j])


# In[ ]:


# populate list of validation sentences
for i in range (0, num_authors):
    for j in range (test_len, (valid_len + test_len)):
        valid_list[0].append(tokens_list[i * num_entries + j])
        valid_list[1].append(identity_list[i * num_entries + j])


# In[ ]:


# populate list of training sentences
for i in range (num_authors):
    for j in range ((valid_len + test_len), (valid_len + test_len + train_len)):
        train_list[0].append(tokens_list[i * num_entries + j])
        train_list[1].append(identity_list[i * num_entries + j])


# In[ ]:


#-----------------------------------------------------------------------------------------------------------------------------#
#-- DEFINITIONS OF MODELS & TRAINING FUNCTIONS--------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#


# In[ ]:


#lots of 5-dim convolutions

class P1_CNN(nn.Module):
    def __init__(self):
        super(P1_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 6, stride=1)
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 6, stride=1)
        self.conv3 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 6, stride=1)
        self.conv4 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 6, stride=1)
        self.conv5 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 6, stride=1)
        self.conv6 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 6, stride=1)
        
        self.pool1 = nn.MaxPool2d(kernel_size = 4, stride= 4)

        self.linear1 = nn.Linear(100, 25, False)
        self.linear2 = nn.Linear(25, 5, False)
        
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        # should I relu pooling layers?
        x = F.relu(self.conv1(x.type(torch.FloatTensor)))
        x = F.relu(self.conv2(x.type(torch.FloatTensor)))
        x = F.relu(self.conv3(x.type(torch.FloatTensor)))
        x = F.relu(self.conv4(x.type(torch.FloatTensor)))
        
        x = self.pool1(x.type(torch.FloatTensor))
        
        x = F.relu(self.conv5(x.type(torch.FloatTensor)))
        x = F.relu(self.conv6(x.type(torch.FloatTensor)))

        x = x.type(torch.FloatTensor).flatten()
        x = F.relu(self.linear1(x.type(torch.FloatTensor)))
        x = F.relu(self.linear2(x.type(torch.FloatTensor)))
        
        return self.softmax(x)


# In[ ]:


CNN_1 = P1_CNN()


# In[ ]:


CNN_1(torch.rand(1,1,100,100))


# In[ ]:


# this CNN takes tensor(1,1,100,100) input
# conv-pool, conv-pool, both small, then linear layers
class P2_CNN(nn.Module):
    def __init__(self):
        super(P2_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 7, stride= 1)
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 8, stride=1)
        
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride= 2)
        self.pool2 = nn.AvgPool2d(kernel_size = 4, stride= 4)
        
        self.linear1 = nn.Linear(100, 25, False)
        self.linear2 = nn.Linear(25, 5, False)
        
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # should I relu pooling layers?
        x = F.relu(self.conv1(x.type(torch.FloatTensor)))
        x = self.pool1(x.type(torch.FloatTensor))
        x = F.relu(self.conv2(x.type(torch.FloatTensor)))
        x = self.pool2(x.type(torch.FloatTensor))
        x = self.dropout(x)
        
        x = x.type(torch.FloatTensor).flatten()
        
        x = F.relu(self.linear1(x.type(torch.FloatTensor)))
        x = F.relu(self.linear2(x.type(torch.FloatTensor)))
        return self.softmax(x)


# In[ ]:


CNN_2 = P2_CNN()


# In[ ]:


CNN_2(torch.rand(1,1,100,100))


# In[ ]:


# this CNN takes tensor(1,1,100,100) input
# one large convolution, one large pooling, then flatten and output
class P3_CNN(nn.Module):
    def __init__(self):
        super(P3_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 31, stride= 1)
        
        self.pool1 = nn.AvgPool2d(kernel_size = 7, stride= 7)
        
        self.linear1 = nn.Linear(100, 25, False)
        self.linear2 = nn.Linear(25, 5, False)
        
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # should I relu pooling layers?
        x = F.relu(self.conv1(x.type(torch.FloatTensor)))
        x = self.pool1(x.type(torch.FloatTensor))
        x = self.dropout(x)
        
        x = x.type(torch.FloatTensor).flatten()
        x = F.relu(self.linear1(x.type(torch.FloatTensor)))
        x = F.relu(self.linear2(x.type(torch.FloatTensor)))
        return self.softmax(x)


# In[ ]:


CNN_3 = P3_CNN()


# In[ ]:


CNN_3(torch.rand(1,1,100,100))


# In[ ]:


# this CNN takes tensor(1,1,100,100) input
# lots of convolutional layers, then pool at the end
class P4_CNN(nn.Module):
    def __init__(self):
        super(P4_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 10, stride= 1)
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 10, stride= 1)
        self.conv3 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 10, stride= 1)
        self.conv4 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 10, stride= 1)
        self.conv5 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 10, stride= 1)
        self.conv6 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 6, stride= 1)
        
        self.pool1 = nn.AvgPool2d(kernel_size = 5, stride= 5)
        
        self.linear1 = nn.Linear(100, 25, False)
        self.linear2 = nn.Linear(25, 5, False)
        
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        # should I relu pooling layers?
        x = F.relu(self.conv1(x.type(torch.FloatTensor)))
        x = F.relu(self.conv2(x.type(torch.FloatTensor)))
        x = F.relu(self.conv3(x.type(torch.FloatTensor)))
        x = F.relu(self.conv4(x.type(torch.FloatTensor)))
        x = F.relu(self.conv5(x.type(torch.FloatTensor)))
        x = F.relu(self.conv6(x.type(torch.FloatTensor)))
        
        x = self.pool1(x.type(torch.FloatTensor))
        
        x = x.type(torch.FloatTensor).flatten()
        x = F.relu(self.linear1(x.type(torch.FloatTensor)))
        x = F.relu(self.linear2(x.type(torch.FloatTensor)))
        return self.softmax(x)


# In[ ]:


CNN_4 = P4_CNN()


# In[ ]:


CNN_4(torch.rand(1,1,100,100))


# In[ ]:


def simple_train(data, model, loss_fn, optimizer, i):
    
    pred = model(data[0][i].type(torch.FloatTensor))
    
    loss = loss_fn(pred, data[1][i].type(torch.FloatTensor))

    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    return(pred, loss)


# In[ ]:


def simple_train2(model, tokens, labels, loss_fn, optimizer2, i):
    
    pred = model(tokens[i].type(torch.FloatTensor))
    
    loss = loss_fn(pred, labels[i].type(torch.FloatTensor))
    
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
        
    if i% 10000 == 0:
        print (i)
    return(pred, loss)


# In[ ]:


shuffle_together([1,2,3,4,5],[1,2,3,4,5])


# In[ ]:


# this function shuffles a list of tokens and shuffles the list of identifying labels in the same way so that calling indices of
# the output retains the connection between label and token.

def shuffle_together(tokens, labels):
    to_shuffle = list(zip(tokens, labels))
    rand.shuffle(to_shuffle)

    tokens, labels = zip(*to_shuffle)
    
    return(tokens, labels)


# In[ ]:


def validate (model, token_list, label_list):
      # randomize the order of data
        token_list, label_list = shuffle_together(token_list, label_list)
        
        accuracy = 0
        list_size = len(token_list)
        
        for j in range(0,list_size):
            
            if torch.argmax(model(token_list[j])) == torch.argmax(label_list[j]):
                accuracy += 1
                
        
        return(accuracy/list_size)


# In[ ]:


def test(model, token_list, label_list):
      # randomize the order of data
        token_list, label_list = shuffle_together(token_list, label_list)
        
        accuracy = 0
        list_size = len(token_list)
        
        for j in range(0,list_size):
            
            if torch.argmax(model(token_list[j])) == torch.argmax(label_list[j]):
                accuracy += 1
                
        
        return(accuracy/list_size)


# In[ ]:


def training_run(model, num_epochs, train_fxn, token_list, label_list, valid_token, valid_label, loss_fxn, optimizer):
    
    output = [[],[]]
    list_size = len(token_list)
    loss_list = []
    train_accuracy_list = []
    valid_accuracy_list = []
    
    for  epoch in range(num_epochs):
        
        # randomize the order of data
        token_list, label_list = shuffle_together(token_list, label_list)
        valid_token, valid_label = shuffle_together(valid_token, valid_label)
        
        
        # reset the accuracy report 
        accuracy = 0
        epoch_loss = 0

        for j in range(0, list_size):
    
            prediction, loss = train_fxn(model, token_list, label_list, loss_fxn, optimizer, j)
            output[0].append(prediction)
            output[1].append(loss)
            
            epoch_loss += loss
            
            if torch.argmax(prediction) == torch.argmax(label_list[j]):
                accuracy +=1
                
                
        loss_list.append(epoch_loss)
        valid_accuracy_list.append(validate(model,valid_token, valid_label))
        train_accuracy_list.append(accuracy/list_size)   
                                                
        print(accuracy/list_size)
        print(epoch, 'done')
        
    return output, loss_list, train_accuracy_list, valid_accuracy_list
        


# In[ ]:


def testing_run(model, token_list, label_list):
      # randomize the order of data
        token_list, label_list = shuffle_together(token_list, label_list)
        
        accuracy = 0
        list_size = len(token_list)
        
        for j in range(0,list_size):
            
            if torch.argmax(model(token_list[j])) == torch.argmax(label_list[j]):
                accuracy += 1
                
        
        return(accuracy/list_size)


# In[ ]:


from scipy.signal import savgol_filter
import matplotlib.pyplot as plt



# In[ ]:


#-----------------------------------------------------------------------------------------------------------------------------#
#-- MODEL INSTANTIATION,OPTIMIZERS, LOSS FUNCTIONS----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#


# In[ ]:


lr_list = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
lr = lr_list[7]
# lr = 1e-7 worked best

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(CNN_1.parameters(), lr = lr)


# In[ ]:


#-----------------------------------------------------------------------------------------------------------------------------#
#--CNN 1----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#


# In[ ]:


# define model and set it to train
CNN_1 = P1_CNN()

CNN_1.train(True)


# In[ ]:


lr_list = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
lr = lr_list[7]
# lr = 1e-7 worked best

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(CNN_1.parameters(), lr = lr)

output1 = training_run(CNN_1, 10, simple_train2, train_list[0], train_list[1], loss_fn, optimizer)


# In[ ]:


smoothed1 = savgol_filter(output1[1], 501, 3)

plt.plot(output1[1])
plt.plot(smoothed1, color='green')
plt.show()


# In[ ]:


CNN_1.train(False)
validate(CNN_1, valid_list[0], valid_list[1])


# In[ ]:


#-----------------------------------------------------------------------------------------------------------------------------#
#--CNN 2----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#


# In[ ]:


# define model and set it to train
CNN_2 = P2_CNN()

CNN_2.train(True)


# In[ ]:


lr_list = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
lr = lr_list[5]
# lr = 1e-7 worked best

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(CNN_2.parameters(), lr = lr)


output2 = training_run(CNN_2, 5, simple_train2, train_list[0], train_list[1], loss_fn, optimizer)


# In[ ]:


accuracy_CNN_2 = [0.20, 0.2195, 0.2332, 0.2508, 0.2587]


# In[ ]:


plt.plot(accuracy_CNN_2)


# In[ ]:


smoothed2 = savgol_filter(output2[1], 501, 3)

plt.plot(output2[1])
plt.plot(smoothed2, color='green')
plt.show()


# In[ ]:


CNN_2.train(False)
validate(CNN_2, valid_list[0], valid_list[1])


# In[ ]:


#-----------------------------------------------------------------------------------------------------------------------------#
#--CNN 3----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------#


# In[ ]:


# define model and set it to train
CNN_3 = P3_CNN()

CNN_3.train(True)


# In[ ]:


loss_list = []
prediction_list = []
epoch_loss_list = []
train_acc_list = []
valid_acc_list = []


# In[ ]:


lr_list = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
lr = lr_list[4]
# lr = 1e-7 worked best


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(CNN_3.parameters(), lr = lr)

output3, epoch_loss, train_acc, valid_acc = training_run(CNN_3, 5, simple_train2, train_list[0], train_list[1], valid_list[0], valid_list[1], loss_fn, optimizer)


# include this after additional trainings
loss_list += output3[0]
prediction_list += output3[1]
epoch_loss_list += epoch_loss
train_acc_list += train_acc
valid_acc_list += valid_acc


# In[ ]:


# save model parameters
torch.save(CNN_3.state_dict(), "D:\a_School Things\Super Senior (Fall 2021)\Winter 2022\LIGN 167\Final Project\CNN_saves")


# In[ ]:


# load model parameters when ready to test again
CNN_3 = P3_CNN()
CNN_3.load_state_dict(torch.load("D:\a_School Things\Super Senior (Fall 2021)\Winter 2022\LIGN 167\Final Project\CNN_saves"))
CNN_3.eval()


# In[ ]:


# set train to false when finished

CNN_3.train(False)


# In[ ]:


# calculate test accuracy
test(CNN_3, test_list[0], test_list[1])


# In[ ]:


## plot loss per epoch and training vs validation accuracy

x_axis = [i+1 for i in range(len(loss_list))]

plt.plot(x_axis, loss_list, label='training loss')
plt.title('Loss for each epoch')
plt.legend();
plt.show()

plt.plot(range(0,5), train_acc, label='training accuracy')
plt.plot(range(0,5), valid_acc, label='validation accuracy')
plt.title('Accuracy for each epoch')
plt.legend();
plt.show()


# In[ ]:


CNN_3.train(True)
output3_1 = training_run(CNN_3, 3, simple_train2, train_list[0], train_list[1], loss_fn, optimizer)


# In[ ]:


output3_2 = training_run(CNN_3, 4, simple_train2, train_list[0], train_list[1], loss_fn, optimizer)


# In[ ]:


smoothed3 = savgol_filter(output3[1], 501, 3)

plt.plot(output3[1])
plt.plot(smoothed3, color='green')
plt.show()


# In[ ]:


smoothed3 = savgol_filter(output3_1[1], 501, 3)

plt.plot(output3_1[1])
plt.plot(smoothed3, color='green')
plt.show()


# In[ ]:


output3_1[1][0:5] + output3[1][0:2]


# In[ ]:


print(min(output3[1][0:1000]), min(output3[1][80000:81000]))


# In[ ]:


CNN_3.train(False)
validate(CNN_3, valid_list[0], valid_list[1])


# In[ ]:





# In[ ]:




