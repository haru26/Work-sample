# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 21:02:36 2020

@author: licor
"""

class surprise(object):
    def read_data(self, data, actor_col = "actor_id", item_col = "interested", stirring_col = "interact",):
        from surprise import Dataset
        from surprise import Reader
        
        reader = Reader(rating_scale=(0.5, 10))

        dataset = Dataset.load_from_df(data[[actor_col, item_col, stirring_col]], reader)
        
        return dataset
        
    def tune_param_func(self, param_grid, dataset, model, evaluation = "rmse"):
        from surprise.model_selection import GridSearchCV

        gs = GridSearchCV(model, param_grid, measures=['rmse', 'mae'], cv=3)
        gs.fit(dataset)
        
        best_model = gs.best_estimator[evaluation]
        best_score = gs.best_score[evaluation]
        best_params = gs.best_params[evaluation]
        
        return best_model, best_score, best_params
    
    
    def train(self, model, dataset, test_size=.25, k = 10, threshold = 3.5):
        from surprise.model_selection import train_test_split
        
        trainset, testset = train_test_split(dataset, test_size = test_size)
        model.fit(trainset)
        
        predictions = model.test(testset)
        
        precise_test, recall_test = self.precision_recall_at_k(predictions, k = k, threshold = threshold)
           
        return model, precise_test, recall_test


    def precision_recall_at_k(self, predictions, k, threshold):
        '''Return precision and recall at k metrics for each user.
        Precision at k is the proportion of recommended items in the top-k set that are relevant
        Recall at k is the proportion of relevant items found in the top-k recommendations'''
        
        from collections import defaultdict
       
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))
    
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
            
        precise_test = sum(prec for prec in precisions.values()) / len(precisions)
        recall_test = sum(rec for rec in recalls.values()) / len(recalls)
    
        return precise_test, recall_test
    
    
    
    

        
        
        
        