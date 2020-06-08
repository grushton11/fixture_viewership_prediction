# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import os.path
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import math
from sklearn.ensemble import RandomForestRegressor
from skgarden import RandomForestQuantileRegressor
import datetime
import xgboost as xgb
from functools import reduce
from sklearn.model_selection import GridSearchCV
import pickle
import joblib


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_mfl_teams(territory, input_df):

    df = input_df.copy()
    df['home_contestant'] = ['{}'.format(i) for i in list(zip(df.home_contestant_concat))]
    df['away_contestant'] = ['{}'.format(i) for i in list(zip(df.away_contestant_concat))]
    home_teams = pd.DataFrame(list(df.home_contestant_concat), columns=['Teams'], index=list(df.home_away_date_concat))
    away_teams = pd.DataFrame(list(df.away_contestant_concat), columns=['Teams'], index=list(df.home_away_date_concat))
    mfl_teams = pd.concat([home_teams,away_teams])

    return mfl_teams

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_trained_teams(territory, input_df):

    trained_features_df = input_df
    trained_teams_df = trained_features_df[(trained_features_df['territory'] == territory) & (trained_features_df['feature_type'] == "competitors")]
    trained_teams_list = list(trained_teams_df['features'])
    teams_df = pd.DataFrame(columns=['Teams'])
    teams_df['Teams'] = trained_teams_df['features']

    return  teams_df, trained_teams_list

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_fixtures_with_encoded_teams(mfl_teams, trained_teams_df, trained_teams_list, df):

    team_list_concatenated = pd.concat([mfl_teams,trained_teams_df])

    # encoding the teams and filtering to only teams trained on
    encoded_teams_concat = pd.get_dummies(team_list_concatenated[(team_list_concatenated['Teams'].isin(trained_teams_list))],prefix='',prefix_sep='')

    encoded_teams_concat.index.rename('home_away_date_concat', inplace=True)
    encoded_teams_concat.reset_index(inplace=True)
    encoded_teams_concat_filtered = encoded_teams_concat[encoded_teams_concat['home_away_date_concat'].isin(df['home_away_date_concat'])]

    encoded_teams_concat_filtered.drop_duplicates(inplace=True)
    encoded_teams_concat_filtered_dedupe = pd.DataFrame(encoded_teams_concat_filtered.groupby(['home_away_date_concat']).sum())
    encoded_teams_concat_filtered_dedupe.reset_index(inplace=True)

    fixtures_with_encoded_teams = pd.merge(encoded_teams_concat_filtered_dedupe,df, how='left', on='home_away_date_concat')

    return fixtures_with_encoded_teams

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_other_trained_features(territory, trained_features_df):

    categoricals = [u'str_stage_name', u'str_competition_name', u'str_ruleset_name', u'str_sport_name', u'weekday', u'month',u'time_of_day']

    trained_dummy_features_df = trained_features_df[(trained_features_df['territory'] == territory) & (trained_features_df['feature_type'].isin(categoricals))]
    trained_dummy_features_df = trained_dummy_features_df.reset_index(drop=True)

    return categoricals, trained_dummy_features_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_mfl_feature_values(df, categoricals):

    mfl_feature_dfs_dictionary = {}

    for i in categoricals:

        mfl_x =  pd.DataFrame(list(df[i]), columns=[i], index=list(df.home_away_date_concat))
        mfl_feature_dfs_dictionary[i] = mfl_x

    return mfl_feature_dfs_dictionary

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_trained_feature_lists(territory, categoricals, trained_features_df):

    dictionary_of_feature_dfs = {}
    dictionary_of_feature_lists = {}

    for i in categoricals:
        trained_x_features_df = trained_features_df[(trained_features_df['territory'] == territory) & (trained_features_df['feature_type'] == i)]
        trained_x_features_df = trained_x_features_df.reset_index(drop=True)

        trained_x_df = pd.DataFrame(columns=[i])
        trained_x_df[i] = trained_x_features_df['features']
        trained_x_df[i] = trained_x_df[i].apply(lambda x: str(x).split(i+"_")[1])
        trained_x_list = list(trained_x_df[i])

        dictionary_of_feature_dfs[i] = trained_x_df
        dictionary_of_feature_lists[i] = trained_x_list

    return dictionary_of_feature_dfs, dictionary_of_feature_lists

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def concat_dummmy_encode_and_filter(dictionary_of_mfl_feature_dfs , dictionary_of_trained_feature_dfs, dictionary_of_trained_feature_lists, categoricals, mfl_df):

    list_of_encoded_dfs = []

    for i in categoricals:

        feature_x_concatenated = pd.concat([dictionary_of_mfl_feature_dfs[i], dictionary_of_trained_feature_dfs[i]])
        feature_x_concat_encoded = pd.get_dummies(feature_x_concatenated[(feature_x_concatenated[i].isin(dictionary_of_trained_feature_lists[i]))])
        feature_x_concat_encoded.index.rename('home_away_date_concat', inplace=True)
        feature_x_concat_encoded.reset_index(inplace=True)
        feature_x_concat_encoded_filtered = feature_x_concat_encoded[feature_x_concat_encoded['home_away_date_concat'].isin(mfl_df['home_away_date_concat'])]
        feature_x_concat_encoded_filtered.drop_duplicates(inplace = True)

        list_of_encoded_dfs.append(feature_x_concat_encoded_filtered)

    return list_of_encoded_dfs

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def prepare_for_scoring(df_final_merged):

    game_info_columns = [u'n_users_8week_lag','home_away_date_concat', 'str_tournament_calendar_name',
                         'home_contestant_prop_fan_base','away_contestant_prop_fan_base', 'away_contestant_concat',
                         'home_contestant_concat','fixture_id', 'date','fixture_month','cust_territory','weekday',
                         'month', 'time_of_day', 'str_sport_name', 'str_competition_name', 'away_contestant',
                         'home_contestant', 'str_stage_name','str_ruleset_name', 'home_contestant_pref_calc_date',
                         'away_contestant_pref_calc_date']


    game_info = df_final_merged[game_info_columns]
    df_for_scoring = df_final_merged.drop(game_info_columns,axis=1)
    df_for_scoring = df_for_scoring.fillna(0)
    df_for_scoring['gameday'] = df_for_scoring['gameday'].replace('Unknown', 0)
    df_for_scoring['gameday'] = pd.to_numeric(df_for_scoring['gameday'])

    return df_for_scoring, game_info

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def load_models(territory_filename, folder):
    print(territory_filename)
    model_folder = dataiku.Folder(folder)
    folder_info = model_folder.get_info()
    path = model_folder.get_path()
    xgb_filename = os.path.join(path,'{}_xgb_model.joblib.compressed'.format(territory_filename))
    xgb_model = joblib.load(xgb_filename)
    return xgb_model

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def score_models(territory, df, xgb_model):
    start_time = datetime.datetime.now()
    print('Starting to score the {} XGBoost model'.format(territory))
    predictions = pd.Series(xgb_model.predict(df),name = 'predictions')
    return predictions

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def post_processing(territory, df, prediction_calculation_date):

    df['dazn_only'] = df['str_tournament_calendar_name'].str.contains('DAZN', na = False)
    now = datetime.datetime.now()
    df['date_prediction_created'] = prediction_calculation_date
    df['model_version'] = "v2.0"
    df['viewership_prediction'] = round(df['predictions'] * df['n_users_8week_lag'])
    df['recommended_broadcast_tier'] = df.apply(lambda df: get_broadcast_tier(territory,df), axis=1)

    return df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def get_broadcast_tier(territory, df):

    if territory == 'DACH' :
        if df['predictions'] >= 0.2:
            return 5
        elif df['predictions'] >= 0.1:
            return 4
        elif df['predictions'] >= 0.025:
            return 3
        elif df['predictions'] >= 0.005:
            return 2
        else: return 1

    elif territory == 'Canada' :
        if df['predictions'] >= 0.25:
            return 5
        elif df['predictions'] >= 0.1:
            return 4
        elif df['predictions'] >= 0.025:
            return 3
        elif df['predictions'] >= 0.005:
            return 2
        else:
            return 1

    elif territory == 'Japan' :
        if df['predictions'] >= 0.15:
            return 5
        elif df['predictions'] >= 0.05:
            return 4
        elif df['predictions'] >= 0.0075:
            return 3
        elif df['predictions'] >= 0.0025:
            return 2
        else:
            return 1

    elif territory == 'Spain' :
        if df['predictions'] >= 0.25:
            return 5
        elif df['predictions'] >= 0.10:
            return 4
        elif df['predictions'] >= 0.05:
            return 3
        elif df['predictions'] >= 0.01:
            return 2
        else: return 1

    elif territory == 'Brazil' :
        if df['predictions'] >= 0.15:
            return 5
        elif df['predictions'] >= 0.05:
            return 4
        elif df['predictions'] >= 0.02:
            return 3
        elif df['predictions'] >= 0.01:
            return 2
        else: return 1

    elif territory == 'United States' :
        if df['predictions'] >= 0.20:
            return 5
        elif df['predictions'] >= 0.05:
            return 4
        elif df['predictions'] >= 0.02:
            return 3
        elif df['predictions'] >= 0.01:
            return 2
        else: return 1

    elif territory == 'Italy' :
        if df['predictions'] >= 0.2:
            return 5
        elif df['predictions'] >= 0.05:
            return 4
        elif df['predictions'] >= 0.02:
            return 3
        elif df['predictions'] >= 0.01:
            return 2
        else: return 1


    else: return

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN

def create_fvp_predictions(cust_territory, territory_filename, mfl_fixtures_df, features_df, model_folder, current_prediction_calculation_date):

    predictions_df = pd.DataFrame([])
    predictions_list = []

    # Return mfl teams df
    mfl_teams = get_mfl_teams(cust_territory, mfl_fixtures_df)

    # Return trained teams df
    trained_teams_df, trained_teams_list = get_trained_teams(cust_territory, features_df)

    # Return a df of fixtures with team features encoded
    fixtures_with_encoded_teams = get_fixtures_with_encoded_teams(mfl_teams,
                                                                  trained_teams_df,
                                                                  trained_teams_list,
                                                                  mfl_fixtures_df)

    # Check if there are any fixtures we can predict upon, else exit
    if fixtures_with_encoded_teams.size >= 1:
        print('There is atleast 1 fixture we can predit upon (after encoding and filtering out less than <10 teams): CONTINUE')
    else:
        print('No encoded fixtures to predict on: EXIT')
        sys.exit(0)

     # Return a df of the categorical features trained on
    categoricals, trained_dummy_features_df = get_other_trained_features(cust_territory, features_df)

    # Return dfs and lists of the categorical features trained on
    trained_feature_dfs_dictionary, trained_feature_lists_dictionary = get_trained_feature_lists(cust_territory, categoricals, features_df)

    # Return dfs of the categorical feature values found in the MFL
    mfl_feature_dfs_dictionary = get_mfl_feature_values(mfl_fixtures_df, categoricals)

    # Take the dictionaries of trained and mfl features, then concat, encode & filter & return a list of dfs which can be merged together
    list_of_encoded_features_dfs = concat_dummmy_encode_and_filter(mfl_feature_dfs_dictionary, trained_feature_dfs_dictionary, trained_feature_lists_dictionary, categoricals, mfl_fixtures_df)

    # Merge the encoded feature dfs together
    df_final_merged = fixtures_with_encoded_teams
    for i in range(len(list_of_encoded_features_dfs)):
        df_final_merged = pd.merge(df_final_merged, list_of_encoded_features_dfs[i], how='left', on= ['home_away_date_concat'])

     # Get the final df ready for scoring
    df_for_scoring, game_info = prepare_for_scoring(df_final_merged)

    # Unpickle the models
    print(territory_filename)
    xgb_model = load_models(territory_filename, model_folder)

    # Score the data against each saved model - note the same ordering of the features in the training data is required
    predictions = score_models(cust_territory, df_for_scoring[features_df['features'].values], xgb_model)

    # Join predictions to the game info
    scored_fixtures_df = game_info.join(predictions)

    # Post processing ready for output
    scored_fixtures_df = post_processing(cust_territory, scored_fixtures_df, current_prediction_calculation_date)

    return scored_fixtures_df
