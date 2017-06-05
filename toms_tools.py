#My tools for data analysis v 1.0
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error



#dummies
def replace_dummies(dataFrame, col, full_name=True, NaN_col = True):
    #replaces a data frames specified col with dummy cols
    try:
        if NaN_col:
            try:
                dataFrame[col].fillna('Nan', inplace=True)
            except:
                pass
        col_loc = dataFrame.columns.get_loc(col)
        dums = pd.get_dummies(dataFrame[col])
        col_names = []
        if full_name:
            for name in dums.columns:
                col_names.append('{}({})'.format(col, name))
            dums.columns = col_names
        df = pd.concat((dataFrame.iloc[:,0:col_loc],dums),axis=1)
        df = pd.concat((df, dataFrame.iloc[:,col_loc+1:]),axis=1)
    except:
        print('Failed to replace {} in df'.format(col))
        df = dataFrame
    finally:
        return df

#define variable exploration function
def is_catagorical(series):
    try:
        if series.describe()['unique']:
            return True
        else:
            return False
    except:
        return False

def get_catagorical_dict(df):
    cat_table = {}
    for col in df.columns:
        cat_table[col] = is_catagorical(df[col])
    return cat_table
def uni_analysis(df, verbose=1):
    is_catagorical_table ={}
    print('Features are :')
    print(df.columns)
    for var in df.columns:
        print('-------------------------------------------------')
        print('                     {}                          '.format(var))
        print('-------------------------------------------------')
       
        feature = df[var]
        print('Feature breakdown: ')
        print(feature.describe())
        print('nulls : {}'.format(feature.isnull().sum()))
        if is_catagorical(feature):
            is_catagorical_table[var] = True
            print(feature.value_counts(normalize=True)*100)
            fig = plt.plot()
            feature.value_counts().plot(kind='bar')
        else:
            is_catagorical_table[var] = False
            ax1 = plt.subplot2grid((2,2), (0,0), rowspan=1, colspan=1)
            ax2 = plt.subplot2grid((2,2), (0,1), rowspan=1, colspan=1)
            ax3 = plt.subplot2grid((2,2), (1,0), rowspan=1, colspan=2)
            feature.plot(ax=ax1, kind='hist')
            feature.plot(ax=ax2, kind='box')
            sns.kdeplot(feature, bw=0.5, ax=ax3)
        plt.show()
    return is_catagorical_table


def my_anova(df, cat_var, cont_var):
    #Dependencies
    #from scipy import stats
    from scipy import stats
    uniques = [x for x in df[cat_var].unique() if str(x) != 'nan']
    anova_list = {}
    for var in uniques:
        anova_list[var] = [x for x in df[df[cat_var] == var][cont_var].values if str(x) != 'nan']
    F, p = stats.f_oneway(*anova_list.values())
    print('F = {}'.format(F))
    print('p = {}'.format(p))

def my_multi_box(df, var_cat, var_cont):
    
    df_two_var=df[[var_cat, var_cont]]
    df_two_var= df_two_var.pivot(columns=df_two_var.columns[0], index = df_two_var.index)
    df_two_var.columns = df_two_var.columns.droplevel()
    if len(df_two_var.columns) > 25:
        print('To many cols')
        return
    #print('debug: cols {}'.format(df_two_var.columns))
    for val in df_two_var.columns:
        percent_str = round((df[df[var_cat]==val].shape[0]/df[var_cat].shape[0])*100,2)
        df_two_var.rename(columns={val:"{} ({})".format(val, percent_str)}, inplace=True)
    df_two_var.boxplot()
    plt.xticks(rotation=90)
    plt.show()    

def multi_analysis(df, cat_table):
    #Prints comparisons of variables and returns a table of which vars are related
    for ix, var1 in enumerate(df.columns):
        for ij, var2 in enumerate(df.columns):
            if ij > ix:
                print("------------------------------------")
                print("{} compared to {}".format(var1, var2))
                print("------------------------------------")
                if cat_table[var1] is True:
                    if cat_table[var2] is True:
                        #cat vs cat
                        print('Catagorical vs Catagorical\n')
                        print("{}\t\t{}".format(var1, var2))
                        for val in filter(lambda x: str(x) != 'nan', train[var1].unique()):
                            
                            group = train[train[var1]==val]
                            print('{} %{:.0f}'.format(val, (group.shape[0]/train.shape[0])*100))
                            for status in filter(lambda x : str(x) != 'nan', train[var2].unique()):
                                subgroup = group[group[var2]==status]
                                print('\t\t {} %{:.0f}'.format(status, (subgroup.shape[0]/group.shape[0])*100))
                                
                        group = df.groupby([var1, var2])
                        group.size().unstack().plot(kind='bar',stacked=True)
                        plt.show()
                    else:
                        print('Catagorical vs Continuous')
                        my_anova(df, var1, var2)
                        my_multi_box(df, var1, var2)
                else:
                    if cat_table[var2] is True: 
                        print('Catagorical vs Continuous')
                        my_anova(df, var2, var1)
                        my_multi_box(df, var2, var1)
                    else:
                        print('Continous vs Continous')
                        print('Correlation: {}'.format(df[var1].corr(df[var2])))
                        plt.scatter(df[var1], df[var2])
                        plt.show()


#roll everything into one function
def knn_impute(data_frame, is_cat, cols_to_impute=[], value_impute=False, val_to_impute_over=False, verbose=False):
    """
        Takes a data frame and imputes values in the given columns using knn
        
        Dependencies:
            sklearn.preprocessing.MinMaxScaler
            sklearn.neighbors.KNeighborsClassifier
            univarient_analysis()
                is_catagorical()
            replace_dummies()
    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    for col_to_imp in cols_to_impute:
        # set catagorical flag
        if is_cat[col_to_imp]:
            cat_flag = True
        else:
            cat_flag = False
        #debug
        print("Cat flag is {}".format(cat_flag))
    #step 1 save nan index and cut out ys
        try:
            if value_impute:
                idx_of_Null = data_frame.loc[data_frame[col_to_imp]==val_to_impute_over].index
                print("imputing_over {}".format(val_to_impute_over))
            else:
                idx_of_Null = pd.isnull(data_frame[col_to_imp]).nonzero()[0]
                print("imputing over nulls")
            features = data_frame.drop(col_to_imp,1)
        except:
            print("{} is not in data_frame maybe?".format(col_to_imp))
            continue
        
    #step 2 fill in nans in X cata and cont
        try:
            #cata
            cat_col = [key for key, val in is_cat.items() if val == True]
            for col in cat_col:
                if col != col_to_imp:
                    features = replace_dummies(features, col)
            #cont
            features.fillna(features.mean(), inplace=True)
            if verbose:
                print(features.head())
        except:
            print("{} having problems filling nans and making dummie cols".format(col_to_imp))
            continue
       
     #step 3 rescale values
        try:
            scaler = MinMaxScaler()
            features_scaled = features.copy()
            cols_to_scale = [keys for keys,val in is_cat.items() if ((val == False) and (keys != col_to_imp))]
            features_scaled[cols_to_scale] = scaler.fit_transform(features_scaled[cols_to_scale])
            if verbose:
                print(features_scaled.head())
        except:
            print("{} problem scaling features see step 3".format(col_to_imp))
            continue
        
    #step 4 create training and sets that need prediction
        try:
            X = pd.concat([data_frame[col_to_imp],features_scaled], axis=1)
            if verbose:
                print(X.head())
            print("ids of nulss")
            print(idx_of_Null)
            to_pred = X.iloc[idx_of_Null]
            to_pred = to_pred.drop(col_to_imp,1)
            if (value_impute):
                X_train = X.drop(X.index[[idx_of_Null]])
            else:
                X_train = X.dropna(subset=[col_to_imp])
            if cat_flag:
                #encode y_vals to one hot
                enc = pd.get_dummies(X_train[col_to_imp])
                col_keys = enc.columns
                y_train = enc.values
            else:
                preds_scaler = MinMaxScaler().fit(X_train[col_to_imp])
                y_train = preds_scaler.transform(X_train[col_to_imp])
            #y_train = X_train[col_to_imp].map({'Female':1, 'Male':0})
            X_train.drop(col_to_imp,1,inplace=True)
            X_train.head()
        except Exception as e:
            print("{} Something went wrong in set creation".format(col_to_imp))
            print(e);
            continue
        
    #step5 train model
        try:
            if cat_flag:
                knn = KNeighborsClassifier(n_neighbors = 7, p=2, metric='minkowski')
            else:
                knn = KNeighborsRegressor(n_neighbors = 7, weights='distance')
            knn.fit(X_train, y_train)
        except:
            print("{} failed to train model".format(col_to_imp))
            continue
    
    #step 6 check accurcay/go back and tune 
        pred_train = knn.predict(X_train)
        if cat_flag:
            print("Training Accuracy: {}".format(accuracy_score(y_train, pred_train)))
        else:
            print("Training MSE: {}".format(mean_squared_error(y_train, pred_train)))
    #step 7 make predictions for values that are missing
        actual_preds = knn.predict(to_pred)
        if verbose:
            print("predictions:")
            print(actual_preds)
    #step 8 replace predictions in model
        if cat_flag:
            #decode from onehot
            decode = pd.DataFrame(actual_preds, columns=col_keys)
            decoded_preds = []
            for idx, row in decode.iterrows():
                for col in decode:
                    if decode.iloc[idx, decode.columns.get_loc(col)] > 0:
                        decoded_preds.append(col)
            decoded_preds= pd.DataFrame(np.array(decoded_preds), columns=[col_to_imp])
            
            for idx, val in zip(idx_of_Null, decoded_preds[col_to_imp].values):
                data_frame.iloc[idx, data_frame.columns.get_loc(col_to_imp)]= val
        else:
            inverse_scaled_preds = preds_scaler.inverse_transform(actual_preds)
            #debug
            print("decoded preds {}".format(inverse_scaled_preds))
            for idx, val in zip(idx_of_Null, inverse_scaled_preds):
                data_frame.iloc[idx, data_frame.columns.get_loc(col_to_imp)]= val
        """
        preds = []
        for val in actual_preds:
            if val == 1:
                preds.append('Female')
            else:
                preds.append('Male')
        """
        
    #return the dataFrame
    return data_frame