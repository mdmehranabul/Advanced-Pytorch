#%%
import numpy as np
from scipy import linalg
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import seaborn as sns
from collections import Counter
# %%
X,y=make_blobs(n_samples=10000,n_features=2, centers=5, random_state=1)
# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# %%
sns.scatterplot(x=X_train[:,0],y=X_train[:,1],hue=y_train)
# %%
INPUT_SIZE=X_train.shape[1]
HIDDEN_SIZE=100
# %%
input_weights=np.random.normal(size=[INPUT_SIZE,HIDDEN_SIZE])
biases=np.random.normal(size=[HIDDEN_SIZE])
# %%
def relu(x):
    return np.maximum(x,0)

def hidden_nodes(X):
    G=np.dot(X,input_weights)
    G=G+biases
    H=relu(G)
    return H

def predict(X):
    out=hidden_nodes(X)
    out=np.dot(out,beta)
    return out
# %%
beta=np.dot(linalg.pinv(hidden_nodes(X_train)),y_train)
# %%
y_test_pred=predict(X_test)
correct=0
total=X_test.shape[0]

for i in range(total):
    predicted=np.round(y_test_pred[i],0)
    y_test_true=y_test[i]
    correct+=1 if predicted==y_test_true else 0
accuracy=correct/total
print(f"Accuracy for {HIDDEN_SIZE} hidden nodes : {accuracy}")

# %% Baseline Accuracy
cnt=Counter(y_test)
max(list(cnt.values()))/total
# %%
