# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import os.path
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from matplotlib import pyplot as plt
import math
from sklearn.ensemble import RandomForestRegressor
from skgarden import RandomForestQuantileRegressor
import datetime
import xgboost as xgb
from functools import reduce
import sklearn
from sklearn.model_selection import GridSearchCV
import pickle
import joblib
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import (r2_score, explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error)
import eli5
from eli5 import show_weights
from eli5.sklearn import PermutationImportance
from sklearn.feature_extraction import DictVectorizer
from matplotlib      import pyplot as plt
from IPython.display import display

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def dummify_teams(df):

    # Create a data frame containing a list of home and away teams
    home = pd.DataFrame(list(df.str_home_contestant_name_concat), columns=['team'], index=list(df.home_away_date_concat))
    away = pd.DataFrame(list(df.str_away_contestant_name_concat), columns=['team'], index=list(df.home_away_date_concat))
    teams = pd.concat([home,away])

    # encode teams
    encoded_teams = pd.get_dummies(teams['team'])
    encoded_teams.index.rename('home_away_date_concat', inplace=True)
    encoded_teams.reset_index(inplace=True)

    # group by fixture
    encoded_teams = pd.DataFrame(encoded_teams.groupby(['home_away_date_concat']).sum())

    # filter teams to >10 appearances
    for i in encoded_teams.columns[1:]:
        if sum(encoded_teams[i]) < 10:
            encoded_teams.drop(i, axis=1, inplace=True)
    encoded_teams.reset_index(inplace=True)

    # merge team features back to the original data frame of fixtures
    out = pd.merge(df, encoded_teams, how='left', on='home_away_date_concat')

    # Get a list of team names
    teams_list = list(encoded_teams.columns.values)
    del teams_list[0]
    teams_output = pd.DataFrame(teams_list)
    teams_output.columns = ['Teams']

    return out

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def create_train_test_splits(df):

    # fill empty values
    df['gameday'] = df['gameday'].replace('Unknown', 0)
    df = df.fillna(0)

    game_info = [u'n_users_8week_lag','home_away_date_concat', 'str_tournament_calendar_name', 'n_users',
                    'fixture_views_sum','home_contestant_prop_fan_base','away_contestant_prop_fan_base',
                   'total_teams_prop_fan_base','str_away_contestant_country','str_home_contestant_country',
                    'str_away_contestant_name','str_home_contestant_name', 'str_away_contestant_name_concat',
                    'str_home_contestant_name_concat','str_fixture_id', 'str_fixture_date','str_fixture_month',
                     'cust_territory', 'P_views']

    game_info_variables = df[game_info]

    categorical_variables = [u'str_stage_name', u'str_competition_name', u'str_ruleset_name', u'str_sport_name', u'weekday', u'month',u'time_of_day']

    numerical_variables = ['larger_team_prop-fan_base','smaller_team_prop-fan_base',u'gameday',
                           u'home_league_position',u'away_league_position',u'str_live_max', u'str_catch_up_max']

    non_competitor_var_size  = game_info_variables.columns.size + len(categorical_variables) + len(numerical_variables) + 1

    competitors = df.columns[non_competitor_var_size:]

    all_columns = categorical_variables.copy()
    all_columns.extend(competitors)
    all_columns.extend(numerical_variables)

    df = df[all_columns]

    df_dummies = pd.get_dummies(df, columns=categorical_variables)
    df = game_info_variables.join(df_dummies)

    # Create dataframe containing all features
    model_features = pd.DataFrame(np.array(list(df_dummies)))
    model_features.columns = ['features']
    for i in competitors:
        model_features.loc[model_features['features'].str.contains(i), 'feature_type'] = 'competitors'

    for i in categorical_variables:
        model_features.loc[model_features['features'].str.contains(i), 'feature_type'] = i

    for i in numerical_variables:
        model_features.loc[model_features['features'].str.contains(i), 'feature_type'] = 'numericals'
        df[i] = df[i].astype('double')

    # Create train/test splits
    train, test = train_test_split(df, test_size=0.2)
    print ('Train data has %i rows and %i columns' % (train.shape[0], train.shape[1]))
    print ('Test data has %i rows and %i columns' % (test.shape[0], test.shape[1]))

    # Apply AVG-STD re-scaling to numerical features
    rescale_features = {u'smaller_team_prop-fan_base': u'AVGSTD',u'larger_team_prop-fan_base': u'AVGSTD'}
    pd.options.mode.chained_assignment = None

    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'AVGSTD':
            shift = train[feature_name].mean()
            scale = train[feature_name].std()
            print ('Rescaled %s' % feature_name)
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

    # Split out the target and predictor variables
    X_train = train.drop(game_info, axis=1)
    y_train = train['P_views']
    X_test = test.drop(game_info, axis=1)
    y_test = test['P_views']

    return train, test, X_train, y_train, X_test, y_test, model_features

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def GridSearch_table_plot(grid_clf, param_name,
                          num_results=15,
                          negative=True,
                          display_all_params=True):

    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    if negative:
        clf_score = -grid_clf.best_score_
    else:
        clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    # display the top 'num_results' results
    # =====================================
    display(pd.DataFrame(cv_results) \
            .sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]
