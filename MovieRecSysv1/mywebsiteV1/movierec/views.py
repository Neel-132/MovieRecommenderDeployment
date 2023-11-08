from django.shortcuts import render
import torch_geometric
import torch
import pickle
import sentence_transformers
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
import numpy as np
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, to_hetero,TransformerConv

# Create your views here.
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = TransformerConv((-1, -1), hidden_channels,heads=7,dropout=0.4)
        self.conv2 = TransformerConv((-1, -1), hidden_channels,heads = 4,dropout=0.2)
        self.conv3 = TransformerConv((-1, -1), out_channels,dropout=0.2)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return z_dict,self.decoder(z_dict, edge_label_index)


def getFeatures(movies_df,ratings_df):
    # One-hot encode the genres:
    genres = movies_df['genres'].str.get_dummies('|').values
    genres = torch.from_numpy(genres).to(torch.float)

    # Load the pre-trained sentence transformer model and encode the movie titles:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with torch.no_grad():
        titles = model.encode(movies_df['title'].tolist(), convert_to_tensor=True, show_progress_bar=True)
        titles = titles.cpu()

    # Concatenate the genres and title features:
    movie_features = torch.cat([genres, titles], dim=-1)
    
    
    user_features = torch.ones(len(ratings_df['userId'].unique()),1)
    
    return user_features,movie_features

def generate_mapped_movieid(movies_list):
    
    movieidsrated = []
    for movie in movies_list:
      movieidsrated.append(list(movies_df.loc[movies_df["title"] == movie]["movieId"])[0])
    mappedmovieid = []

    for id in movieidsrated:
      mappedmovieid.append(list(unique_movie_id.loc[unique_movie_id["movieId"] == id]["mappedMovieId"])[0])
        
    return mappedmovieid 


'''def createGraph(user_features,movie_features,ratings_df,movies_df):
    
    data = HeteroData()

    # Add the user nodes:
    data['user'].x = user_features  # [num_users, num_features_users]

    # Add the movie nodes:
    data['movie'].x = movie_features  # [num_movies, num_features_movies]

    # Add the rating edges:
    data['user', 'rates', 'movie'].edge_index = edge_index  # [2, num_ratings]

    # Add the rating labels:
    rating = torch.from_numpy(ratings_df['rating'].values).to(torch.float)
    data['user', 'rates', 'movie'].edge_label = rating  # [num_ratings]

    # We also need to make sure to add the reverse edges from movies to users
    # in order to let a GNN be able to pass messages in both directions.
    # We can leverage the `T.ToUndirected()` transform for this from PyG:
    data = T.ToUndirected()(data)

    # With the above transformation we also got reversed labels for the edges.
    # We are going to remove them:
    del data['movie', 'rev_rates', 'user'].edge_label
    
    return data '''

def createnewgraph(mappedmovieid,ratings_list):
    
    new_id = torch.max(edge_index[0,:]) + 1 #New User ID. To be taken from login
    usernew = new_id * np.ones(len(mappedmovieid)).astype(int)    
    
    ratedmovies = torch.tensor(mappedmovieid)
    add = torch.stack((usernew,ratedmovies))
    
    new_edge_index = torch.cat((edge_index,add),-1)
    
    rating = torch.from_numpy(ratings_df['rating'].values).to(torch.float)
    new_ratings = torch.cat((rating,torch.tensor(ratings_list)),dim = 0)
    
    new_user_features = torch.ones(user_features.size()[0] + 1,1)
    
    newdata = HeteroData()

    # Add the user nodes:
    newdata['user'].x = new_user_features  # [num_users, num_features_users]

    # Add the movie nodes:
    newdata['movie'].x = movie_features  # [num_movies, num_features_movies]

    # Add the rating edges:
    newdata['user', 'rates', 'movie'].edge_index = new_edge_index  # [2, num_ratings]

    # Add the rating labels:
    newdata['user', 'rates', 'movie'].edge_label = new_ratings  # [num_ratings]

    # We also need to make sure to add the reverse edges from movies to users
    # in order to let a GNN be able to pass messages in both directions.
    # We can leverage the `T.ToUndirected()` transform for this from PyG:
    newdata = T.ToUndirected()(newdata)

    # With the above transformation we also got reversed labels for the edges.
    # We are going to remove them:
    del newdata['movie', 'rev_rates', 'user'].edge_label
    
    return newdata,int(new_id)


def save_graph(graph, path):
    with open(path, 'wb') as f:
        pickle.dump(graph, f)
    print("Graph saved successfully")
    f.close()

def load_graph(path):
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    f.close()

    return graph

def getinfo(data):
    tup = data.splitlines()
    movie = []
    ratings = []
    for el in tup:
        l = el.strip().split(',')
        print(l) 
        movie.append(l[0].strip())
        ratings.append(float(l[1].strip()))
    return movie,ratings

def load_trainedmodel(path):
    global data
    data = load_graph("graph.pkl")
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(hidden_channels=100).to(device)
    checkpoint = torch.load(path,map_location = torch.device("cpu"))
    model.load_state_dict(checkpoint)

    return model

def createnewgraph(mappedmovieid,ratings_list):
    edge_index = data['user', 'rates', 'movie'].edge_index
    user_features = data['user'].x
    movie_features = data['movie'].x


    new_id = torch.max(edge_index[0,:]) + 1 #New User ID. To be taken from login
    usernew = new_id * np.ones(len(mappedmovieid)).astype(int)    
    
    ratedmovies = torch.tensor(mappedmovieid)
    add = torch.stack((usernew,ratedmovies))
    
    new_edge_index = torch.cat((edge_index,add),-1)
    
    rating = torch.from_numpy(ratings_df['rating'].values).to(torch.float)
    new_ratings = torch.cat((rating,torch.tensor(ratings_list)),dim = 0)
    
    new_user_features = torch.ones(user_features.size()[0] + 1,1)
    
    newdata = HeteroData()

    # Add the user nodes:
    newdata['user'].x = new_user_features  # [num_users, num_features_users]

    # Add the movie nodes:
    newdata['movie'].x = movie_features  # [num_movies, num_features_movies]

    # Add the rating edges:
    newdata['user', 'rates', 'movie'].edge_index = new_edge_index  # [2, num_ratings]

    # Add the rating labels:
    newdata['user', 'rates', 'movie'].edge_label = new_ratings  # [num_ratings]

    
    newdata = T.ToUndirected()(newdata)

   
    del newdata['movie', 'rev_rates', 'user'].edge_label
    
    return newdata,int(new_id)

def runnewdata(newdata,model):
    with torch.no_grad():
        newdata = newdata.to(device)
        embeds,pred = model(newdata.x_dict, newdata.edge_index_dict,
                     newdata['user', 'movie'].edge_index)
        pred = pred.clamp(min=0, max=5)
        target = newdata['user', 'movie'].edge_label.float()
        
    return embeds

def movies_user_user(embeds,new_id):
    new_user_embeds = embeds["user"][new_id,:]
    cosine_similarities = F.cosine_similarity(new_user_embeds, embeds["user"], dim=1)
    top_similarities, top_indices = torch.topk(cosine_similarities, k=20)
    top_indices.tolist()
    similar_users = []
    for i in top_indices.tolist()[1:]: #mapped to userid
      similar_users.append(i+1)
    similar_users
    joineddf = ratings_df.merge(movies_df,on="movieId")
    moviesdf = joineddf.loc[joineddf["userId"].isin(similar_users[1:])] #What similar users have watched
    movieset = set(moviesdf.loc[moviesdf["rating"]==5]["title"])
    moviesetids = set(moviesdf.loc[moviesdf["rating"]==5]["mappedMovieId"]) #For movies similar to ones watched by user in moviest
    return movieset

def movies_item_item(embeds,mappedmovieid):
    movie_indexes=[]
    for i in embeds["movie"][mappedmovieid,:]:
        cosine_similarities_movie = F.cosine_similarity(i, embeds["movie"], dim=1)
        top_similarities, top_indices = torch.topk(cosine_similarities_movie, k=30)
        movie_indexes.extend(top_indices.tolist())
    
    list(set(movie_indexes))
    return list(movies_df.merge(unique_movie_id,on="movieId").iloc[list(set(movie_indexes)),:]["title"])

def movies_item_user(embeds,new_id):
    new_user_embeds = embeds["user"][new_id,:]
    cosine_similarities = F.cosine_similarity(new_user_embeds, embeds["user"], dim=1)
    top_similarities, top_indices = torch.topk(cosine_similarities, k=100)
    
    return list(movies_df.merge(unique_movie_id,on="movieId").iloc[top_indices.tolist(),:]["title"])

def RecommendMovies(request):
    global movies_df
    global ratings_df
    global unique_user_id
    global unique_movie_id
    movies_df = pd.read_csv("movies.csv") 
    ratings_df = pd.read_csv("ratings.csv")
    del ratings_df["timestamp"]
    movies_df = movies_df.loc[movies_df["movieId"].isin(ratings_df['movieId'].unique())]

    unique_user_id = ratings_df['userId'].unique() 
    unique_user_id = pd.DataFrame(data={
        'userId': unique_user_id,
        'mappedUserId': pd.RangeIndex(len(unique_user_id))
        })
    unique_movie_id = movies_df['movieId'].unique()
    unique_movie_id = pd.DataFrame(data={
        'movieId': unique_movie_id,
        'mappedMovieId': pd.RangeIndex(len(unique_movie_id))
        })
        
    ratings_df = ratings_df.merge(unique_user_id, on='userId')
    ratings_df = ratings_df.merge(unique_movie_id, on='movieId')


    
    movies_list,ratings_list = getinfo(request.POST["movies"])

    model = load_trainedmodel("TransformerConv.pt")

    mappedmovieid = generate_mapped_movieid(movies_list)
    
    newdata,new_id = createnewgraph(mappedmovieid,ratings_list) 

    embeds = runnewdata(newdata,model)
    
    l1 = movies_user_user(embeds,new_id)
    l2 = movies_item_item(embeds,mappedmovieid)
    l3 = movies_item_user(embeds,new_id)

    #return render(request,"RecomInput.html",{'l1': l1,'l2': l2,'l3': l3})
    return render(request,"RecomOutput.html",{'l1': l1,'l2': l2,'l3': l3})

def home(request):
	return render(request,"RecomInput.html")




