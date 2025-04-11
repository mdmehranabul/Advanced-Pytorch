import pandas as pd 
from sklearn import model_selection,preprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import mean_squared_error

df=pd.read_csv('ratings.csv')
print(df.head(2))

print(f"Unique Users {df.userId.nunique()}, Unique Movies: {df.movieId.nunique()}")
device=torch.device("cpu")
class MovieDataset(Dataset):
    def __init__(self,users,movies,ratings) -> None:
        super().__init__()
        self.users=users
        self.movies=movies
        self.ratings=ratings
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self,idx):
        users=self.users[idx]
        movies=self.movies[idx]
        ratings=self.ratings[idx]
        return torch.tensor(users,dtype=torch.long),torch.tensor(movies,dtype=torch.long),torch.tensor(ratings,dtype=torch.float)
    
class RecSysModel(nn.Module):
    def __init__(self,n_users,n_movies,n_embeddings=32):
        super().__init__()
        self.user_embed=nn.Embedding(n_users,n_embeddings)
        self.movie_embed=nn.Embedding(n_movies,n_embeddings)
        self.out=nn.Linear(n_embeddings*2,1)
    
    def forward(self,users,movies):
        user_embeds=self.user_embed(users)
        movie_embeds=self.movie_embed(movies)
        x=torch.cat([user_embeds,movie_embeds],dim=1)
        x=self.out(x)
        return x

lbl_user=preprocessing.LabelEncoder()
lbl_movie=preprocessing.LabelEncoder()
df['userId']=lbl_user.fit_transform(df['userId'])
df['movieId']=lbl_movie.fit_transform(df['movieId'])

# Create train test dataset

df_train,df_test=model_selection.train_test_split(df,test_size=0.3,random_state=123)

train_dataset=MovieDataset(users=df_train.userId.values,
                           movies=df_train.movieId.values,
                           ratings=df_train.rating.values)

valid_dataset=MovieDataset(users=df_test.userId.values,
                           movies=df_test.movieId.values,
                           ratings=df_test.rating.values)

print("Sample from dataset:")
print(train_dataset[0])

# DataLoaders
BATCH_SIZE=32
train_loader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
test_loader=DataLoader(dataset=valid_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
print("Dataloaders created")

model=RecSysModel(n_users=len(lbl_user.classes_)+1,n_movies=len(lbl_movie.classes_)+1)
model.to(device)
optimizer=torch.optim.Adam(model.parameters())
criterion=nn.MSELoss()

NUM_EPOCHS=1

model.train()
for epoch_i in range(NUM_EPOCHS):
    print(f"  → Epoch {epoch_i} begin")
    for users,movies,ratings in train_loader:
        optimizer.zero_grad()
        y_pred = model(users, movies)         
        y_true = ratings.unsqueeze(dim=1).to(torch.float32)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

print("Training complete!")

y_preds=[]
y_trues=[]

model.eval()

with torch.no_grad():
    for users,movies,ratings in test_loader:
        y_true = ratings.detach().numpy().tolist()
        y_pred = model(users, movies).squeeze().detach().numpy().tolist()
        y_trues.extend(y_true)
        y_preds.extend(y_pred)

mse=mean_squared_error(y_trues,y_preds)
print(f"Mean Squarted Error:{mse}")

# Users and Items
user_movie_test=defaultdict(list)

with torch.no_grad():
    for users,movies,ratings in test_loader:
        y_true = ratings.detach().numpy().tolist()
        y_pred = model(users, movies)
        for i in range(len(users)):
            user_id=users[i].item()
            movie_id=movies[i].item()
            pred_rating=y_pred[i][0].item()
            true_rating=ratings[i].item()
            print(f"User: {user_id}, Movie: {movie_id}, Pred: {pred_rating}, True:{true_rating}")
            user_movie_test[user_id].append((pred_rating,true_rating))

precisions={}
recalls={}

k=10
thres=3.5

for uid,user_ratings in user_movie_test.items():
    user_ratings.sort(key = lambda x: x[0], reverse=True )
    
    # count of relevant items
    n_rel=sum((rating_true>=thres) for _, rating_true in user_ratings)
    
    # count of recommended items that are predicted and within top k
    n_rec_k = sum((rating_pred>=thres) for rating_pred,_ in user_ratings[:k])
    
    # count recommended and relevant items
    n_rel_and_rec_k=sum((rating_pred>=thres) and (rating_true>=thres) for rating_pred,rating_true in user_ratings[:k])
    
    print(f"Uid:{uid}, n_rel:{n_rel} n_rec_k:{n_rec_k} n_rel_and_rec_k  {n_rel_and_rec_k}")
    precisions[uid]=n_rel_and_rec_k/n_rec_k if n_rec_k !=0 else 0
    recalls[uid]=n_rel_and_rec_k/n_rel if n_rel!=0 else 0

print(f"Precisions@{k} : {sum(precisions.values())/len(precisions)}")
print(f"Recalls@{k} : {sum(recalls.values())/len(recalls)}")
          
    
    
    