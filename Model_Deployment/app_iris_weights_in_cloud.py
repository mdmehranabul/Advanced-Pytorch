#%%
from flask import Flask,request
from model_class import MultiClassNet
import torch
import torch.nn as nn
import json
import requests
#%% Download model weights
URL='https://storage.googleapis.com/iris-model-bert/model_iris.pt'
r=requests.get(URL)
local_file_path='model_iris_from_gcp.pt'
with open(local_file_path,'wb')as f:
    f.write(r.content)
    f.close()
#%%
model=MultiClassNet(NUM_FEATURES=4, NUM_CLASSES=3, HIDDEN_FEATURES=6)
# local_file_path='model_iris.pt'
model.load_state_dict(torch.load(local_file_path))
#%%
app=Flask(__name__)

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return 'Please use POST method'
    if request.method=='POST':
        data=request.data.decode('utf-8')
        dict_data = request.get_json(force=True)
        X=torch.tensor([dict_data["data"]])
        y_test_hat_softmax=model(X)
        y_test_hat=torch.max(y_test_hat_softmax,1)
        y_test_cls=y_test_hat.indices.cpu().detach().numpy()[0]
        class_dict={
            0:'setosa',
            1:'versicolor',
            2:'virginica'
        }
    return f"Your flower belongs to class {class_dict[y_test_cls]}"

if __name__=='__main__':
    app.run()

# %%
