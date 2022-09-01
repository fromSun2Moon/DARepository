import pandas as pd
import random
import os,sys
import numpy as np
from itertools import combinations

import pickle
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

########################
#모든 모델 앙상블
########################
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor,\
                                    BaggingRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LinearRegression, HuberRegressor, BayesianRidge , ElasticNet
from sklearn.ensemble import VotingRegressor
# from sklearn.model_selection import cross_validate

#################
# Environ settings
#################
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def validation(gt, preds):
    # NRMSE
    def lg_nrmse(gt, preds):
        # 각 Y Feature별 NRMSE 총합
        # Y_01 ~ Y_08 까지 20% 가중치 부여
        all_nrmse = []
        for idx in range(14): # ignore 'ID'
            rmse = mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
            nrmse = rmse/np.mean(np.abs(gt[:,idx]))
            all_nrmse.append(nrmse)
        score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
        return score
    valid_score = lg_nrmse(gt.values, preds)
    return valid_score


def save_model(model, fname):
    msg = "[INFO] File Saved...."
    pickle.dump(model, open(fname, 'wb'))
    print(msg)

def loaded_model(fname):
    msg = "[INFO] File Loaded..."
    model = pickle.load(open(fname, 'rb'))
    print(msg)
    return model 

def test_and_save(model, submit,test_x, save_dir):
    preds = model.predict(test_x)
    for idx, col in enumerate(submit.columns):
        if col=='ID':
            continue
        submit[col] = preds[:,idx-1]
    submit.to_csv(save_dir, index=False)



if __name__ == "__main__":
    seed = 1006
    #####################
    # 콤비네이션 개수 
    #####################
    r = int(sys.argv[1])
    seed_everything(1006) # Seed 고정
    # Machine Learning Regression 
    
    RGs = {"rfc":RandomForestRegressor(random_state=seed), "lgb" : LGBMRegressor(random_state=seed), "lr":LinearRegression(),\
            "xgb": XGBRegressor(), "kn": KNeighborsRegressor(), "det": DecisionTreeRegressor(random_state=seed), "adb": AdaBoostRegressor(random_state=seed),\
            "ex" : ExtraTreesRegressor(random_state=seed), "hb":HuberRegressor(),"bsr": BayesianRidge(), "eln": ElasticNet(random_state=seed),\
            "rdg": Ridge(random_state=seed),"bg": BaggingRegressor(random_state=seed), "gb": GradientBoostingRegressor(random_state=seed)}
    
    #######################
    # 저장할 장소 지정
    #######################
    root_dir = "./auto_csv_result"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Generate combinations
    combinations_reg = list(combinations(RGs.keys(), r=r))
    combination=[]
    if r == 6:
        for a,b,c,d,e,f in combinations_reg:
            combination.append([(a , RGs[a]), (b , RGs[b]), (c , RGs[c]), (d, RGs[d]), (e, RGs[e]),(f,RGs[f])])
    elif r == 5:
        for a,b,c,d,e in combinations_reg:
            combination.append([(a , RGs[a]), (b , RGs[b]), (c , RGs[c]), (d, RGs[d]), (e, RGs[e])])
        # weights = [0.9,0.8,0.7,0.6,0.5]
    elif r == 3:
        for a,b,c in combinations_reg:
            combination.append([(a , RGs[a]), (b , RGs[b]), (c , RGs[c])])
        weights = [1,0.9,0.8]
    elif r == 2:
        for a,b in combinations_reg:
            combination.append([(a, RGs[a]), (b, RGs[b])])

    else:
        raise Exception("This code is only training for combinations of machine learning models!") 

    ################
    # Data settings
    ################
    train_df = pd.read_csv('./train.csv')
    train_set = train_df.sample(frac=0.85,random_state=1006)
    val_set = train_df.iloc[list(set(train_df.index) - set(train_set.index))]
    train_x = train_set.filter(regex='X') # Input : X Featrue
    train_y = train_set.filter(regex='Y') # Output : Y Feature
    val_x = val_set.filter(regex='X') 
    val_y = val_set.filter(regex='Y') 

    test_x = pd.read_csv('./test.csv').drop(columns=['ID'])
    submit = pd.read_csv('./sample_submission.csv')

    best_score = 10000.
    scores=[]
    for i in tqdm(range(len(combination))):
        combi_names = '_'.join([combination[i][x][0] for x in range(len(combination[i]))])
        
        models = combination[i]
        print(models)
        models  = MultiOutputRegressor(VotingRegressor(models), n_jobs=-1) # setting multi-processing
        models.fit(train_x, train_y) # training
        
        # validation process
        val_preds = models.predict(val_x)
        score = validation(val_y, val_preds)
        
        print(f"[INFO] score : {score} | combinations : {combi_names}")
        print()
        scores.append(score)
        
            
        if best_score > score:
            best_score = score
            fname = root_dir + f"/best_reg_models_" + combi_names + str(score)[:7] + ".pkl"
            save_dir = root_dir +"/ensems_" + combi_names + "_" + str(score) + ".csv"
            print(f"[INFO] best score : {best_score} | combinations : {combi_names}")
            save_model(models, fname)
            test_and_save(models, submit, test_x, save_dir)
        
        # re-initialize varibles
        del models
        del val_preds
        del score
