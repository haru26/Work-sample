# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:16:03 2020

@author: licor
"""

class MF_deep_learning(object):
    
    def regular(self, df, rating_col = "interact"):
        df["y"] = (df[rating_col]-df[rating_col].mean())/df[rating_col].std()
        return df
    
    def read_data(self, data, user_col = "actor_id", item_col = "interested"):
        import pandas as pd
        
        data_encoded = data.copy()
        
        users = pd.DataFrame(data_encoded[user_col].unique(),columns=[user_col])  # df of all unique users
        dict_users = users.to_dict()    
        inv_dict_users = {v:k for k, v in dict_users[user_col].items()}
    
        items = pd.DataFrame(data_encoded[item_col].unique(),columns=[item_col]) # df of all unique items
        dict_items = items.to_dict()    
        inv_dict_items = {v:k for k, v in dict_items[item_col].items()}
    
        data_encoded.actor_id = data_encoded[user_col].map(inv_dict_users)
        data_encoded.interested = data_encoded[item_col].map(inv_dict_items)
    
        return data_encoded, dict_users, dict_items
    
        
    def model(self, df):
        import tensorflow as tf
        from tensorflow import keras
        
        embedding_size = actor_embedding_size = 8
        actor_id_input = keras.Input(shape=(1,), name='actor_id')
        interested_input = keras.Input(shape=(1,), name='interested')
        
        interested_r12n = keras.regularizers.l2(1e-6)
        actor_r12n = keras.regularizers.l2(1e-7)
        actor_embedded = keras.layers.Embedding(len(df.actor_id.unique()), actor_embedding_size,
                                               embeddings_initializer='glorot_uniform',
                                               embeddings_regularizer=actor_r12n,
                                               input_length=1, name='actor_embedding')(actor_id_input)
        interested_embedded = keras.layers.Embedding(len(df.interested.unique()), embedding_size, 
                                                embeddings_initializer='glorot_uniform',
                                                embeddings_regularizer=interested_r12n,
                                                input_length=1, name='interested_embedding')(interested_input)
        
        dotted = keras.layers.Dot(2)([actor_embedded, interested_embedded])
        out = keras.layers.Flatten()(dotted)
        
        l2_model = keras.Model(
            inputs = [actor_id_input, interested_input],
            outputs = out
        )
        
        
        l2_model.compile(
            tf.train.AdamOptimizer(0.005),
            loss='MSE',
            metrics=['MAE', 'MSE'],
        )
        l2_model.summary(line_length=88)
        
        return l2_model
    
    
    def train(self, model, df, batch_size=10**4, epochs=10, verbose=2, validation_split=.05):
        model.fit([df.actor_id, df.interested],
                  df.y,
                  batch_size=10**4,
                  epochs=10,
                  verbose=2,
                  validation_split=.05,
                    )

    def recommend(self, df, model, user_id):
        import pandas as pd
        
        """Return a DataFrame with the n most highly recommended movies for the user with the
        given id. (Where most highly recommended means having the highest predicted ratings 
        according to the given model).
        The returned DataFrame should have a column for movieId and predicted_rating (it may also have
        other columns).
        """
        candidate_actor = pd.DataFrame(df["interested"].unique())
        candidate_actor.columns = ['interested']
        candidate_actor.set_index('interested')
    
        preds = model.predict([
            [user_id] * len(candidate_actor), # User ids 
            candidate_actor.index, # Movie ids
        ])
        # Because our model was trained on a 'centered' version of rating (subtracting the mean, so that
        # the target variable had mean 0), to get the predicted star rating on the original scale, we need
        # to add the mean back in.
    
        mean = df.interact.mean()
        std = df.interact.std()
        candidate_actor['predicted_rating'] = (preds * std + mean)
        candidate_actor["actor_id"] = user_id
        return candidate_actor
    
    def get_recommendation(self, df, model):
        import pandas as pd
        
        pred = pd.DataFrame()
        for i in (df["actor_id"].unique()):
            pred_df = self.recommend(df, model, i)
            pred = pred.append(pred_df, ignore_index=True)
            
        pred_true = pd.merge(df, pred, on = ["interested", "actor_id"])
            
        return pred,  pred_true
    
    def precision_recall_at_k(self, data, k=10, threshold=3.5):
        from collections import defaultdict
    
        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for index, row in data.iterrows():
            user_est_true[row["actor_id"]].append((row["predicted_rating"], row["interact"]))
            
        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
    
            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)
    
            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
    
            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
    
            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                  for (est, true_r) in user_ratings[:k])
    
            # Precision@K: Proportion of recommended items that are relevant
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
    
            # Recall@K: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
            
        precision_mean = sum(prec for prec in precisions.values()) / len(precisions)
        recall_mean = sum(rec for rec in recalls.values()) / len(recalls)
        
        return precision_mean, recall_mean