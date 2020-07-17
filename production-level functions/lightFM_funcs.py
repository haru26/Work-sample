# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:34:16 2020

@author: licor
"""

class LightFM(object):
    
    def read_data(self, df, user_col = 'actor_id', item_col = 'interested', rating_col = 'interact', 
                  norm= False, threshold = None):
        from scipy import sparse
        
        interactions = df.groupby([user_col, item_col])[rating_col] \
                .sum().unstack().reset_index(). \
                fillna(0).set_index(user_col)
        if norm:
            interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
            
        interactions = sparse.csr_matrix(interactions.values)
        
        return interactions
    
    
    def runMF(self, interactions, n_components, learning_rate, loss, k, epoch, n_jobs):
        from lightfm import LightFM
        model = LightFM(no_components= n_components, learning_rate = learning_rate,
                        loss=loss,k=k)
        model.fit(interactions,epochs=epoch,num_threads = n_jobs)
        return model
    
    
    
    def tune_param_func(self, interactions, space, test_percentage = 0.25, random_state = 2020, n_calls = 10, loss = "warp"):
        from lightfm.evaluation import auc_score
        import numpy as np
        from lightfm.cross_validation import random_train_test_split
        from skopt import forest_minimize
        from lightfm import LightFM
        
        def objective(params):
            # unpack
            epochs, learning_rate, no_components = params
        
            model = LightFM(loss=loss,
                            random_state=random_state,
                            learning_rate=learning_rate,
                            no_components=no_components)
            model.fit(train, epochs=epochs,
                      num_threads=4, verbose=True)
        
            patks = auc_score(model, test, num_threads=4)
            maptk = np.mean(patks)
            # Make negative because we want to _minimize_ objective
            out = -maptk
            # Handle some weird numerical shit going on
            if np.abs(out + 1) < 0.01 or out < -1.0:
                return 0.0
            else:
                return out
        
        train, test = random_train_test_split(interactions, test_percentage=test_percentage, random_state=None)
        
        res_fm = forest_minimize(objective, space, n_calls=n_calls,
                             random_state=random_state ,
                             verbose=False)
        max_auc = -res_fm.fun
        
        params = ['epochs', 'learning_rate', 'no_components']
        params_list = []
        for (p, x_) in zip(params, res_fm.x):
            params_list.append((p, x_))
            
        return max_auc, params_list
    
    def train (self, interactions, test_percentage=0.25,
               n_components=30, learning_rate = 0.5, loss='warp', model_k=15, n_jobs = 4, 
               epoch=30, evaluate_k = 50):

        from lightfm.evaluation import precision_at_k
        from lightfm.evaluation import recall_at_k
        from lightfm.cross_validation import random_train_test_split
        
        train, test = random_train_test_split(interactions, test_percentage=test_percentage, random_state=None)
        
        mf_model = self.runMF(interactions = train,
                         n_components = n_components,
                         learning_rate = learning_rate,
                         loss = loss,
                         k = model_k,
                         epoch = epoch,
                         n_jobs = n_jobs)
        
        precise = precision_at_k(mf_model, test_interactions = test, k = evaluate_k)
        recall = recall_at_k(mf_model, test_interactions = test, k = evaluate_k)
        
        precise_test = precise.mean()
        recall_test = recall.mean()
        
        return mf_model, precise_test, recall_test
        