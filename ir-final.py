# -*- coding: utf-8 -*-
"""train-simple-xray-cnn.ipynb
# Goal
The goal is to use a simple model to classify x-ray images in Keras, the notebook how to use the ```flow_from_dataframe``` to deal with messier datasets
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
# %matplotlib inline
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from IPython.core.debugger import set_trace

# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda:0') if USE_CUDA else 'cpu'
print("CUDA:", USE_CUDA)
print(DEVICE)
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.set_default_dtype(torch.float64)


all_xray_df = pd.read_csv('C:/input/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('C:', 'input', 'images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(x[:-1]))
all_xray_df.sample(3)

"""# Preprocessing Labels
Here we take the labels and make them into a more clear format. The primary step is to see the distribution of findings and then to convert them to simple binary labels
"""

label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 90)

all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
from itertools import chain
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df.sample(3)

"""### Clean categories
Since we have too many categories, we can prune a few out by taking the ones with only a few examples
"""

# keep at least 1000 cases
MIN_CASES = 1000
all_labels = [c_label for c_label in all_labels if all_xray_df[c_label].sum()>MIN_CASES]
print('Clean Labels ({})'.format(len(all_labels)), 
      [(c_label,int(all_xray_df[c_label].sum())) for c_label in all_labels])

# since the dataset is very unbiased, we can resample it to be a more reasonable collection
# weight is 0.1 + number of findings
sample_weights = all_xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
sample_weights /= sample_weights.sum()
all_xray_df = all_xray_df.sample(40000, weights=sample_weights)

label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 90)

label_counts = 100*np.mean(all_xray_df[all_labels].values,0)
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
ax1.set_xticklabels(all_labels, rotation = 90)
ax1.set_title('Adjusted Frequency of Diseases in Patient Group')
_ = ax1.set_ylabel('Frequency (%)')

"""# Prepare Training Data
Here we split the data into training and validation sets and create a single vector (disease_vec) with the 0/1 outputs for the disease status (what the model will try and predict)
"""

all_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])

from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(all_xray_df, 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = all_xray_df['Finding Labels'].map(lambda x: x[:4]))
print('train', train_df.shape[0], 'validation', valid_df.shape[0])

"""# Create Data Generators
Here we make the data generators for loading and randomly transforming images
"""

from keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (128, 128)
core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen

train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 32)

valid_gen = flow_from_dataframe(core_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 256) # we can use much larger batches for evaluation
# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next(flow_from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 1024)) # one big batch

t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -1.5, vmax = 1.5)
    c_ax.set_title(', '.join([n_class for n_class, n_score in zip(all_labels, c_y) 
                             if n_score>0.5]))
    c_ax.axis('off')

"""# Create a simple model
Here we make a simple model to train using MobileNet as a base and then adding a GAP layer (Flatten could also be added), dropout, and a fully-connected layer to calculate specific features
"""

from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
base_mobilenet_model = MobileNet(input_shape =  t_x.shape[1:], 
                                 include_top = False, weights = None)
multi_disease_model = Sequential()
multi_disease_model.add(base_mobilenet_model)
multi_disease_model.add(GlobalAveragePooling2D())
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(512))
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(len(all_labels), activation = 'sigmoid'))
multi_disease_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy', 'mae'])
multi_disease_model.summary()

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=3)
callbacks_list = [checkpoint, early]

"""# First Round
Here we do a first round of training to get a few initial low hanging fruit results
"""

multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch=100,
                                  validation_data = (test_X, test_Y), 
                                  epochs = 1, 
                                  callbacks = callbacks_list)

"""# Check Output
Here we see how many positive examples we have of each category
"""

for c_label, s_count in zip(all_labels, 100*np.mean(test_Y,0)):
    print('%s: %2.2f%%' % (c_label, s_count))

pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)

"""# ROC Curves
While a very oversimplified metric, we can show the ROC curve for each metric
"""

from sklearn.metrics import roc_curve, auc
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('barely_trained_net.png')

"""# Continued Training
Now we do a much longer training process to see how the results improve and saving locally
"""

multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch = 100,
                                  validation_data =  (test_X, test_Y), 
                                  epochs = 5, 
                                  callbacks = callbacks_list)

# load the best weights
multi_disease_model.load_weights(weight_path)

pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)

# look at how often the algorithm predicts certain diagnoses 
for c_label, p_count, t_count in zip(all_labels, 
                                     100*np.mean(pred_Y,0), 
                                     100*np.mean(test_Y,0)):
    print('%s: Dx: %2.2f%%, PDx: %2.2f%%' % (c_label, t_count, p_count))

from sklearn.metrics import roc_curve, auc
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('trained_net.png')

"""# Show a few images and associated predictions"""

sickest_idx = np.argsort(np.sum(test_Y, 1)<1)
fig, m_axs = plt.subplots(4, 2, figsize = (16, 32))
for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):
    c_ax.imshow(test_X[idx, :,:,0], cmap = 'bone')
    stat_str = [n_class[:6] for n_class, n_score in zip(all_labels, 
                                                                  test_Y[idx]) 
                             if n_score>0.5]
    pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  for n_class, n_score, p_score in zip(all_labels, 
                                                                  test_Y[idx], pred_Y[idx]) 
                             if (n_score>0.5) or (p_score>0.5)]
    c_ax.set_title('Dx: '+', '.join(stat_str)+'\nPDx: '+', '.join(pred_str))
    c_ax.axis('off')
fig.savefig('trained_img_predictions.png')

"""#Calculate Bleu Score """
data = 'Chest-X-ray'
video_vector = np.load('C:/Dataset/'+ data + '/videoFeatures.npy',allow_pickle=True)
video_vector = video_vector.item()
train_videos = np.load('C:/Dataset/'+ data + '/train_videos.npy')
train_videos = [train_videos[i].item() for i in range(len(train_videos))]
train_captions = np.load('C:/Dataset/'+ data + '/train_captions.npy')
train_captions = [train_captions[i].item() for i in range(len(train_captions))]

test_videos = np.load('C:/Dataset/' + data + '/test_videos.npy')
test_videos = [test_videos[i].item() for i in range(len(test_videos))]
test_captions = np.load('C:/Dataset/' + data + '/test_captions.npy')
test_captions = [test_captions[i].item() for i in range(len(test_captions))]

len(train_videos),len(train_captions), len(test_videos),len(test_captions)

import spacy
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
en = spacy.load('en')
EN_TEXT = Field(init_token='<sos>',
           eos_token='<eos>',
           tokenize=lambda captions : [ [tok.text for tok in en.tokenizer(sentence)] for sentence in captions],
           batch_first = True)
EN_TEXT.build_vocab(EN_TEXT.tokenize(train_captions))
len(EN_TEXT.vocab.stoi)

from collections import defaultdict

train_references = defaultdict(list)
for i in range(len(train_captions)):
  train_references[train_videos[i]].append(train_captions[i].split())

test_references = defaultdict(list)
for i in range(len(test_captions)):
  test_references[test_videos[i]].append(test_captions[i].split())
  
len(train_references), len(test_references)

from torch.utils.data import Dataset
class SampleDataset(Dataset):
  def __init__(self,videoID):
    self.samples = []
    for i in range(len(videoID)):
      if videoID[i] in video_vector :
        self.samples.append(videoID[i])
 
  def __len__(self):
      return len(self.samples)
 
  def __getitem__(self,idx):
      return((self.samples[idx],torch.as_tensor(video_vector[self.samples[idx]],dtype=torch.float64)))

train_dataset = SampleDataset(list(train_references.keys()))
test_dataset = SampleDataset(list(test_references.keys()))

from torch.utils.data import DataLoader
from tqdm import notebook
import torch
train_loader = DataLoader(train_dataset,batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset,batch_size=64, shuffle=True, num_workers=2)

"""### ***Utility Functions***"""

import nltk
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu

def get_sentences(pred):
  sentences = []
  for i in range(pred.shape[0]):
    sentence = []
    for j in range(pred.shape[1]):
      if pred[i,j] == EN_TEXT.vocab.stoi['<eos>'] :
        break
      sentence.append(EN_TEXT.vocab.itos[pred[i,j].item()])
    sentences.append(' '.join(sentence))
  return sentences

def evaluate(inp, max_length):
  enc_hidden = model.encoder.initialize_hidden_state(inp.shape[0])
  enc_output, hidden = model.encoder(inp, enc_hidden)  
  dec_input = torch.full(size = (inp.shape[0],1), fill_value = EN_TEXT.vocab.stoi['<sos>'],device=DEVICE)

  pred = torch.empty(inp.shape[0],max_length,dtype = torch.int64)

  for i in range(max_length):
    predictions, hidden = model.decoder(dec_input, hidden, enc_output)
    output = torch.argmax(predictions,dim = 1)
    dec_input = output.view(inp.shape[0],1)
    pred[:,i] = output.cpu()
    
  return get_sentences(pred)


def get_scores(data_loader, data_references):
  references = []
  candidates = []
  for batch_no, (v,inp) in notebook.tqdm_notebook(enumerate(data_loader)) :
    o = evaluate(inp.to(device = DEVICE),30)
    for i in range(inp.shape[0]):
      l = sum(len(s) for s in data_references[v[i]])//len(data_references[v[i]])
      candidates.append(o[i].split()[:l])
      references.append(data_references[v[i]])
  result = {}
  result['BLEU1'] = corpus_bleu(references, candidates, weights=(1.0, 0, 0, 0))
  return result

"""### ***Load Model***"""

def load_checkpoint(filepath, model):
    checkpoint = torch.load(filepath,map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['state_dict'])
    # print(model.eval())

# Commented out IPython magic to ensure Python compatibility.
import os
model_names = ['Base Model','Proposed Model']

for model_name in model_names :
  print(model_name)
  path = 'C:/' + model_name + '/' + data + '/'

  if model_name == 'Base Model':
#     %run 'C:/Base Model/Model.ipynb'
  if model_name == 'Proposed Model':
#     %run 'C:/Proposed Model/Model.ipynb'

  model = Encoder_Decoder_Model(seq_len, input_size, enc_dim, embedding_dim, vocab_size, dec_units)
  if USE_CUDA :
    model = model.cuda()

  start_epoch = 0
  while os.path.exists(path + 'checkpoint'+str(start_epoch+1) +'.pth'):
    start_epoch += 1
  # start_epoch = 49

  if start_epoch > 0:
    load_checkpoint(path + 'checkpoint'+str(start_epoch)+'.pth', model)
    print('{} Loaded from {} Epoch\n'.format(model_name, start_epoch))
  
  print('Calculating Metric Scores of the {}'.format(model_name))
  test_result = get_scores(test_loader, test_references)
  print('Metric Scores of the {} on the {} Dataset : '.format(model_name, data))
  print(test_result)
  print("\n\n")

import numpy as np
import matplotlib.pyplot as plt
root = 'C:'
data = 'Chest-X-ray'

def plot_multiple(models):
    plt.figure(figsize=(16,8))
    for model in models :
        path = root + model + '/' + data + '/losses.npy'
        losses = np.load(path, mmap_mode="r")
        ls = np.linspace(1,len(losses)+1,len(losses))
        plt.plot(ls,losses,label = model)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
plot_multiple(model_names)

