#%%
import torch
import torchtext.vocab as vocab
# %%
glove=vocab.GloVe(name='6B',dim=100)
# %% number of words & embeddings
glove.vectors.shape
# %% get an embedding vector

def get_embedding_vector(word):
    word_index=glove.stoi[word]
    emb=glove.vectors[word_index]
    return emb
    
get_embedding_vector('chess').shape
# %% find closest words from the input
def get_closest_words_from_word(word,max_n=5):
    word_emb=get_embedding_vector(word)
    distances=[(w,torch.dist(word_emb,get_embedding_vector(w)).cpu().item()) for w in glove.itos]
    dist_sort_filt=sorted(distances,key=lambda x:x[1])[:max_n]
    return dist_sort_filt

get_closest_words_from_word('chess')

# %%
