
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import pickle

# loading dataset and models
data = pd.read_csv('Files/06_prepared_for_ml.csv')

pickle_in_clf_tuned = open('Files/XGBClassifier_tuned.sav', 'rb')
pickle_in_clf = open('Files/XGBClassifier_tuned.sav', 'rb')
pickle_in_reg = open('Files/XGBRegressor_tuned.sav', 'rb')

clf_tuned = pickle.load(pickle_in_clf_tuned)
clf = pickle.load(pickle_in_clf)
reg = pickle.load(pickle_in_reg)

# Metrics

clf_t_prec = round((0.773391 * 100), 2)
clf_t_rec = round((0.249482 * 100), 2)
clf_t_f1 = round((0.375768 * 100), 2)
clf_t_auc = round(0.935852, 2)

clf_prec = round((0.695108 * 100), 2)
clf_rec = round((0.456744 * 100), 2)
clf_f1 = round((0.550971 * 100), 2)
clf_auc = round(0.956500, 2)

reg_rmse = round(0.832025 * 1000000)
reg_ev = round((0.640946 * 100), 2)

# Variables for the app

devs = ['<select>'] + list(np.sort(data['Developer'].unique()))
pubs = ['<select>'] + list(np.sort(data['Publisher'].unique()))

max_sugg = data['Suggest_count'].max()

platforms = list(pd.Series(data.filter(like='P_')
                   .columns)
                   .apply(lambda x: x.split('P_')[1]))

genres = list(pd.Series(data.filter(like='G_')
                            .columns)
                            .apply(lambda x: x.split('G_')[1]))

tags = list(pd.Series(data.filter(like='T_')
                            .columns)
                            .apply(lambda x: x.split('T_')[1]))

current_year = dt.date.today().year

esrbs = ['Unknown', 'Everyone', 'Everyone 10+', 'Teen', 'Mature 17+', 'Adults 18+']

years = ['<select>', 0] + sorted([*range(1971, current_year + 20)], reverse = True)

months = ['<select>', '0', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

nonzero_scores = data['Scores'].apply(lambda x: True if x > 0 else False)
default_score = data[nonzero_scores]['Scores'].median()


#########################################################################################################
# Functions
#########################################################################################################

def encode_input(value, df_ref, is_dev = True):
    
    '''
    Given an input Developer/ Publisher string value and a dataframe of reference, finds its encoded value in said dataset.
    
    value (string): Name of the Developer/ Publisher of interest.
    df_reference : The dataframe from where we are getting the encoded value.
    is_dev: Whether the passed value is Developer or Publisher. Value by default = True.
    '''
    
    if is_dev:
        col = 'Developer'
        enc_col = 'Developer_enc'
        
    else:
        col = 'Publisher'
        enc_col = 'Publisher_enc'
    

    if df_ref[df_ref[col] == value].shape[0] < 1: # Value not in the top 50
        encoded = df_ref[df_ref[col] == 'Other'][enc_col].iloc[0]

    else:
        encoded = df_ref[df_ref[col] == value][enc_col].iloc[0]

    return encoded


#--------------------------------------------------------------------------------------------------------

## We are going to consider followers in social media as a virtual suggest_count, as both things
## represent certain interest on the product.

def get_sug_count(x, limit = 10000):
    
    '''
    Given a number, if it is smaller tha n the limit parameter, it returns the number cast to integer.
    If this number is equal to or greater than limit, then returns limit.
    
    x: Imput number.
    limit: The threshold to cap the number. Default value = 10000.
    '''
    
    x = int(x)
    
    if x < limit:
        out = x
        
    else:
        out = limit
    
    return out

#--------------------------------------------------------------------------------------------------------

def ohe_list_like(x, ref_list):
    
    '''
    Given a list like input, returns another list with 1 where it corresponds to the mode selected in the list and a 0 in the rest.
    
    x: the list of platforms we want to convert.
    '''
    
    out = []
    
    for element in ref_list:

        if element in x:
            out.append(1)
        else:
            out.append(0)
    
    return out

#--------------------------------------------------------------------------------------------------------

def generate_datapoint(publisher, developer, interest, pos_f, neg_f, plats_list, genres_list, tags_list, esrb, release_y, release_m):
    
    '''
    Given the input values, it generates a data point to perform the prediction on with an estimator.
    
    publisher: The name of the publisher.
    developer: The name of the developer.
    sug_count: Suggestion count.
    platforms: A list of platforms the game will be released for.
    genres: A list of the game's genres.
    tags: A list of the game's related tags.
    esrb: The game's intended ESRB rating.
    release_y: The year of release.
    release_m: The month of release.
    '''
    
    plat_ohe = ohe_list_like(plats_list, platforms)
    genre_ohe = ohe_list_like(genres_list, genres)
    esrb_ohe = ohe_list_like(esrb, esrbs)
    tag_ohe = ohe_list_like(tags_list, tags)

    result_dict = {
                    'Suggest_count': get_sug_count(interest),
        
                    'P_MicroSoft': plat_ohe[0],
                    'P_Nintendo': plat_ohe[1],
                    'P_Other': plat_ohe[2],
                    'P_PC': plat_ohe[3],
                    'P_Sony': plat_ohe[4],
        
                    'G_Action': genre_ohe[0],
                    'G_Adventure': genre_ohe[1],
                    'G_Arcade': genre_ohe[2],
                    'G_Casual': genre_ohe[3],
                    'G_Family': genre_ohe[4],
                    'G_Fighting': genre_ohe[5],
                    'G_Indie': genre_ohe[6],
                    'G_Massively_Multiplayer': genre_ohe[7],
                    'G_Platformer': genre_ohe[8],
                    'G_Puzzle': genre_ohe[9],
                    'G_RPG': genre_ohe[10],
                    'G_Racing': genre_ohe[11],
                    'G_Shooter': genre_ohe[12],
                    'G_Simulation': genre_ohe[13],
                    'G_Sports': genre_ohe[14],
                    'G_Strategy': genre_ohe[15],
                    'G_Other': genre_ohe[16],
        
                    'ESRB_All': esrb_ohe[0],
                    'ESRB_10+': esrb_ohe[1],
                    'ESRB_Teen': esrb_ohe[2],
                    'ESRB_17+': esrb_ohe[3],
                    'ESRB_18+': esrb_ohe[4],
        
                    'T_Singleplayer': tag_ohe[0],
                    'T_Multiplayer': tag_ohe[1],
                    'T_Co_Op': tag_ohe[2],
                    'T_Online': tag_ohe[3],
                    'T_Great_OST': tag_ohe[4],
                    'T_Atmospheric': tag_ohe[5],
                    'T_Violent': tag_ohe[6],
                    'T_Story_Rich': tag_ohe[7],
                    'T_2D': tag_ohe[8],
                    'T_Funny': tag_ohe[9],
                    'T_Horror': tag_ohe[10],
                    'T_Retro': tag_ohe[11],
                    'T_Sci_fi': tag_ohe[12],
                    'T_Open_World': tag_ohe[13],
                    'T_1st_Person': tag_ohe[14],
                    'T_3rd_Person': tag_ohe[15],
                    'T_Fantasy': tag_ohe[16],
                    'T_Fem_Protag': tag_ohe[17],
                    'T_Hard': tag_ohe[18],
                    'T_FPS': tag_ohe[19],
        
                    'Release_Y': int(release_y),
                    'Release_M': int(release_m),
                    'Developer_enc': encode_input(developer, data),
                    'Publisher_enc': encode_input(publisher, data, is_dev = False),
                    'Scores': default_score,
                    'Positives': pos_f,
                    'Negatives': neg_f
    }
    
    return pd.DataFrame([result_dict])


#########################################################################################################
# Web app
#########################################################################################################

html_temp = '''
<div style = "background-color:teal; padding:10px">
<h1 style = "color:white; text-align:center;">Video Games Hit Predictor</h1>
</div>
'''
st.markdown(html_temp, unsafe_allow_html=True)

st.write('''
This app estimates how successful a game could be by predicting 
its probability of becoming a hit, and its units sold in the 1st year after release.
''')

selection = st.sidebar.radio('Select the model used to make the prediction.', ['Tuned','Untuned'], index = 0) 
st.sidebar.write(f'''**Tuned**: \n\nHas {clf_t_prec}% Avg Precision/ {clf_t_rec}% Avg Recall/ {clf_t_f1}% Avg F1/ {clf_t_auc} Avg ROC AUC\n\n
**Untuned**: \n\nHas {clf_prec}% Avg Precision/ {clf_rec}% Avg Recall/ {clf_f1}% Avg F1/ {clf_auc} Avg ROC AUC''')


col1, col2 = st.beta_columns([1,2])


# Input variables

with col1:
    title = st.text_input('Game Title:', 'Your video game\'s title')
    release_y = st.selectbox('Release year (select 0 if unknown)', years, index = 0)
    release_m = st.selectbox('Release month (select 0 if unknown)', months)
    esrb_ = st.selectbox('ESRB/ Intended ESRB',esrbs)
    interest = st.number_input('Highest number of followers in social media:', 0)
    pos_feedbk = st.number_input('Amount of positive feedback received',0)
    neg_feedbk = st.number_input('Amount of negative feedback received',0)
    

with col2:
    plats_ = st.multiselect('Platform group/s for launch (you can select multiple values)', platforms)
    genres_ = st.multiselect('Genre (you can select multiple values)', genres)
    tags_ = st.multiselect('Tags related to the game (you can select multiple values)', tags)
    dev = st.selectbox('Developer', devs, index = 0)
    pub = st.selectbox('Publisher', pubs, index = 0)
    button = st.button('Predict')
    result = st.beta_container()

# Submit

if button:
    
    if title == 'Your video game\'s title':
        st.error('Please, enter yout game\'s title.')
    
    try:
        release_y = int(release_y)
        
    except ValueError:
        st.error('Please, enter a valid input for Release year.')

    
    try:
        release_y = int(release_m)
        
    except ValueError:
        st.error('Please, enter a valid input for Release month.')
        
    
    df = generate_datapoint(pub, dev, interest, pos_feedbk, neg_feedbk, plats_, genres_, tags_, esrb_, release_y, release_m)
    
    if selection == 'Tuned':
        clf_prob = round(float(clf_tuned.predict_proba(df)[:,1]) * 100, 2)
        prec = clf_t_prec
        rec = clf_t_rec
    
    else:
        clf_prob = round(float(clf.predict_proba(df)[:,1]) * 100, 2)
        prec = clf_prec
        rec = clf_rec

        
    reg_pred = round(reg.predict(df)[0] * 1000000)
    a = reg.predict(df)
    
    with result:
        st.success(f'**{title}** has a **{clf_prob}**% chance of becoming a hit.')
        st.success(f'Estimated units sold in the 1st year after release: **{reg_pred}**.')
        st.success(f'Root mean squared error: {reg_rmse} units (avg)\n\nExplained variance: {reg_ev}% (avg)\n\nPrecision: {prec}% (avg)\n\nRecall: {rec}% (avg)')
        
        
