import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import ppscore as pps
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_regression,mutual_info_classif

def prepare_data(df):
    
    df = df.rename(columns={col: col.strip() for col in df.columns})
    df = df.dropna(subset=['price'])
    
    df = df.copy()
    df['price'] = df['price'].astype(str)
    df['price'] = df['price'].str.replace('₪', '', regex=True)
    df['price'] = df['price'].str.replace(',', '', regex=True)
    df['price'] = df['price'].str.replace('TOP10', '', regex=True)
    df['price'] = df['price'].str.replace('מתיווך', '', regex=True)
    df['price'] = df['price'].str.replace('בבלעדיות', '', regex=True)
    df['price'] = df['price'].str.replace('במדד המתווכים', '', regex=True)
    df['price'] = df['price'].str.replace('בנה ביתך', '', regex=True)
    df['price'] = df['price'].str.replace(' ', '', regex=True)
    df['price'] = pd.to_numeric(df['price'])
    df = df.dropna(subset=['price'])
    df['price'] = df['price'].astype(int)
    
    df = df.copy()
    df['Area'] = df['Area'].astype(str)
    df['Area'] = df['Area'].str.replace('מ"ר', '', regex=True)
    df['Area'] = df['Area'].str.replace(' ', '', regex=True)
    df['Area'] = df['Area'].str.replace('nan', '', regex=True)
    df['Area'] = df['Area'].str.replace('None', '', regex=True)
    df['Area'] = df['Area'].str.replace('-', '', regex=True)
    df['Area'] = df['Area'].str.replace(')', '', regex=True)
    df['Area'] = df['Area'].str.replace('(', '', regex=True)
    df['Area'] = df['Area'].str.replace('עסקאותבאיזור1000', '', regex=True)
    
    df['Area'] = pd.to_numeric(df['Area'])

    df["City"] = df["City"].str.strip()
    df['City'] = df['City'].astype(str)
    df['City'] = df['City'].str.replace('נהרייה', 'נהריה', regex=True)
    
    df.loc[df["type"] == "מגרש", "floor_out_of"] = np.nan
    df.loc[df["type"] == "נחלה", "floor_out_of"] = np.nan
    df["type"] = df["type"].replace("קוטג'", "קוטג")
    df["type"] = df["type"].replace("קוטג' טורי", "קוטג טורי")
    
    df['room_number'] = df['room_number'].astype(str)

    df["room_number"] = df["room_number"].str.strip()
    
    df['room_number'] = df['room_number'].str.replace(']', '', regex=True)
    df['room_number'] = df['room_number'].str.replace('[', '', regex=True)
    df['room_number'] = df['room_number'].str.replace('חד', '', regex=True)
    df['room_number'] = df['room_number'].str.replace('\'', '', regex=True)
    df['room_number'] = df['room_number'].str.replace('-', '', regex=True)
    df['room_number'] = df['room_number'].str.replace('׳', '', regex=True)
    df['room_number'] = df['room_number'].str.replace('nan', '', regex=True)
    
    df['room_number'] = pd.to_numeric(df['room_number'])
    
    ## Fix the Street column:
    
    def clean_name(name):
        if isinstance(name, str):
            #street = re.sub(r'\([^()]*\)', '', name)
        
            #pattern = re.compile(r'[^\u0590-\u05FF\s"]')
            pattern = re.compile(r'[^\u0590-\u05FF\s]') 
            name = re.sub(pattern, '', name)
            name = name.strip()
            return name
        else:
            return name

    df['Street'] = df['Street'].apply(clean_name)
    
    df['Street'] = df['Street'].fillna('').astype(str)
    df = df[~df['Street'].str.contains('בשכונת')]
    
    ## Fix the number_in_street
    
    df['number_in_street'] = df['number_in_street'].astype(str)

    def convert_to_integer(value):
        try:
            return int(value)
        except ValueError:
            return np.nan 
    
    df["number_in_street"] = df["number_in_street"].apply(convert_to_integer)

    ## Fix the city_area
    
    df['city_area'] = df['city_area'].apply(clean_name)
    df['city_area'] = df['city_area'].astype(str)
    df['city_area'] = df['city_area'].replace('nan', '')
    
    ## Fix the num_of_images
    
    df['num_of_images'] = df['num_of_images'].fillna(0)
    
    def extract_numbers(details):
        if isinstance(details, str):
            numbers = re.findall(r'\d+', details)
            num_count = len(numbers)

            if num_count == 2:
                return int(numbers[0]), int(numbers[1])
            elif num_count == 1:
                return int(numbers[0]), np.nan
    
        return details, np.nan
    
    
    df["floor"], df["total_floors"] = zip(*df["floor_out_of"].map(extract_numbers))

    df["floor"] = df["floor"].replace('קומת קרקע' , 0)
    df["floor"] = df["floor"].replace('קומת מרתף' , -1)
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
    
    df = df[~((df['type'] == 'דירת גן') & (df['floor'] > 2))]
    mask = ((df['type'] == 'קוטג') | (df['type'] == 'קוטג טורי') | (df['type'] == 'דו משפחתי') | (df['type'] == 'בית פרטי')) & ((df['floor'] > 5) | (df['total_floors'] > 5))
    df = df[~mask]
    
    df.loc[(df['type'] == 'בית פרטי') & (~df['floor'].isin([0, np.nan])), 'floor'] = 0
    df.loc[(df['type'] == 'קוטג') & (~df['floor'].isin([0, np.nan])), 'floor'] = 0
    df.loc[(df['type'] == 'קוטג טורי') & (~df['floor'].isin([0, np.nan])), 'floor'] = 0
    df.loc[(df['type'] == 'דו משפחתי') & (~df['floor'].isin([0, np.nan])), 'floor'] = 0
        
    ## fix the T/N columns
    
    df['hasElevator'] = df['hasElevator'].astype(str)
    df["hasElevator"] = df["hasElevator"].replace(["True","יש","יש מעלית", "yes", "כן"], 1)
    df["hasElevator"] = df["hasElevator"].replace(["False","אין","אין מעלית", "no","לא"], 0)
    df["hasElevator"] = pd.to_numeric(df["hasElevator"], errors="coerce")
    
    df['hasParking'] = df['hasParking'].astype(str)
    df["hasParking"] = df["hasParking"].replace(["True","יש חנייה","יש","יש חניה", "yes", "כן"], 1)
    df["hasParking"] = df["hasParking"].replace(["False","אין","אין חניה","אין חנייה", "no","לא"], 0)
    df["hasParking"] = pd.to_numeric(df["hasParking"], errors="coerce")
    
    df['hasBars'] = df['hasBars'].astype(str)
    df["hasBars"] = df["hasBars"].replace(["True","יש","יש סורגים", "yes", "כן"], 1)
    df["hasBars"] = df["hasBars"].replace(["False","אין","אין סורגים", "no","nan","לא"], 0)
    df["hasBars"] = pd.to_numeric(df["hasBars"], errors="coerce")
    
    df['hasStorage'] = df['hasStorage'].astype(str)
    df["hasStorage"] = df["hasStorage"].replace(["True","יש","יש מחסן", "yes", "כן"], 1)
    df["hasStorage"] = df["hasStorage"].replace(["False","אין","אין מחסן", "no","לא"], 0)
    df["hasStorage"] = pd.to_numeric(df["hasStorage"], errors="coerce")
    
    df['hasAirCondition'] = df['hasAirCondition'].astype(str)
    df["hasAirCondition"] = df["hasAirCondition"].replace(["True","יש","יש מיזוג אויר","יש מיזוג אוויר", "yes", "כן"], 1)
    df["hasAirCondition"] = df["hasAirCondition"].replace(["False","אין","אין מיזוג אוויר","אין מיזוג אויר", "no","לא"], 0)
    df["hasAirCondition"] = pd.to_numeric(df["hasAirCondition"], errors="coerce")
    
    df['hasBalcony'] = df['hasBalcony'].astype(str)
    df["hasBalcony"] = df["hasBalcony"].replace(["True","יש","יש מרפסת", "yes", "כן"], 1)
    df["hasBalcony"] = df["hasBalcony"].replace(["False","אין","אין מרפסת", "no","לא"], 0)
    df["hasBalcony"] = pd.to_numeric(df["hasBalcony"], errors="coerce")
    
    df['hasMamad'] = df['hasMamad'].astype(str)
    df["hasMamad"] = df["hasMamad"].replace(["True","יש", "yes", "כן"], 1)
    df["hasMamad"] = df["hasMamad"].replace(["False","אין", "no","לא"], 0)
    
    df.loc[df["hasMamad"].str.contains("יש", na=False), "hasMamad"] = 1
    df.loc[df["hasMamad"].str.contains("אין", na=False), "hasMamad"] = 0
    
    
    df["hasMamad"] = pd.to_numeric(df["hasMamad"], errors="coerce")
    
    df['handicapFriendly'] = df['handicapFriendly'].astype(str)
    df["handicapFriendly"] = df["handicapFriendly"].replace(["True","נגיש לנכים","נגיש","yes", "כן"], 1)
    df["handicapFriendly"] = df["handicapFriendly"].replace(["False","לא נגיש לנכים","לא נגיש","no","לא"], 0)
    
    df["handicapFriendly"] = pd.to_numeric(df["handicapFriendly"], errors="coerce")
        
    ## fix the condition column
    
    df['condition'] = df['condition'].astype(str)

    df["condition"] = df["condition"].replace(["לא צויין","nan", "None","False"], 'not_defind')
    df['condition'] = df['condition'].replace({'משופץ': 'renovated' , 'שמור': 'maintained' , 'חדש': 'new' , 'ישן': 'old' , 'דורש שיפוץ': 'requires_renovation'})

    ## fix the entranceDate column
    
    def categorize_date(date):
        if isinstance(date, datetime):
            now = datetime.now()
            six_months_later = now + timedelta(days=180)
            one_year_later = now + timedelta(days=365)

            if date < six_months_later:
                return "less_than_6 months"
            elif date <= one_year_later:
                return "months_6_12"
            else:
                return "above_year"

        return date
    
    df["entrance_date"] = df["entranceDate"]

    df["entrance_date"] = df["entrance_date"].replace("מיידי", "less_than_6 months")
    df["entrance_date"] = df["entrance_date"].replace("גמיש", "flexible")
    df["entrance_date"] = df["entrance_date"].replace("גמיש ", "flexible")
    df["entrance_date"] = df["entrance_date"].replace("לא צויין", "not_defined")
    
    df["entrance_date"] = df["entrance_date"].apply(categorize_date)
    
    ## fix the furniture column
    
    df["furniture"] = df["furniture"].replace("לא צויין", "not_defined")
    df["furniture"] = df["furniture"].replace("חלקי", "partial")
    df["furniture"] = df["furniture"].replace("אין", "no")
    df["furniture"] = df["furniture"].replace("מלא", "full")
    
    ## fix the publishedDays column
    
    df['publishedDays'] = df['publishedDays'].astype(str)

    df["publishedDays"] = df["publishedDays"].replace("None ", "nan")
    df["publishedDays"] = df["publishedDays"].replace("None", "nan")
    df["publishedDays"] = df["publishedDays"].replace("Nan", "nan")
    df["publishedDays"] = df["publishedDays"].replace("-", "nan")
    df["publishedDays"] = df["publishedDays"].replace("חדש", "0")
    df["publishedDays"] = df["publishedDays"].replace("חדש!", "0")
    
    df["publishedDays"] = pd.to_numeric(df["publishedDays"], errors="coerce")
    
    ## fix the description column
    
    df["description"] = df["description"].str.replace(r"[^A-Za-zא-ת0-9.\"']", " ", regex=True)
    
    ## More repairs
    
    df.loc[(df['room_number'] >= 15) & (df['room_number'] % 5 == 0), 'room_number'] /= 10
    
    df.loc[(df['type'] == 'בניין') & (~df['floor_out_of'].isna()) & (df['floor_out_of'] != ''), 'type'] = 'דירה'
    
    df = df.loc[~((df['type'] == 'דירת גג') & (df['floor_out_of'] == 'קומת קרקע'))]
    
    df.loc[((df['type'] == 'פנטהאוז') | (df['type'] == 'מיני פנטהאוז')) & (df['floor'] <= df['total_floors'] - 2), 'type'] = 'דירה'
    
    df = df.loc[~((df['floor'] > df['total_floors']) & (~df['floor'].isna()) & (~df['total_floors'].isna()))]
    
    df.loc[df['type'].isin(['נחלה', 'מגרש','בניין']), 'room_number'] = 0
    
    df['Area'] = df['Area'].fillna(df.groupby(['type','room_number'])['Area'].transform('mean'))
    
    def round_room_number(value): 
        rounded_value = np.round(value, decimals=1)
        decimal_digit = rounded_value - np.floor(rounded_value)
        if decimal_digit < 0.5:
            return np.floor(rounded_value)
        elif decimal_digit > 0.5:
            return np.ceil(rounded_value)
        else:
            return rounded_value

    
    df['room_number'] = df['room_number'].fillna(df.groupby(['type','Area'])['room_number'].transform('mean'))
    
    df['Area_range'] = np.round(df['Area'] / 50) * 50
    df['room_number'] = df['room_number'].fillna(df.groupby(['type','Area_range'])['room_number'].transform('mean'))
        
    df['room_number'] = df['room_number'].apply(round_room_number)
    
    
    if 'Area_range' in df.columns:
        df.drop(columns=['Area_range'], inplace=True)
        #print("Area_range Column exist in the DataFrame.")
    else:
        print("Area_range Column does not exist in the DataFrame.")
        
    ## Dropping useless column
    
    df = df.drop(columns=['publishedDays'])
    
    ## Dropping duplicates rows
    
    df.drop_duplicates(inplace=True)
    
    ## drop more useless columns
    
    df = df.drop(columns=['description','entranceDate','floor_out_of','number_in_street'])
    
    ## drop rows with nan values
    
    df = df.drop(columns=['total_floors'])
    df = df.replace('', np.nan).dropna()
    
    df = df.dropna()
    
    ## change the price to be the last column:
        
    moved_column = df['price']
    df = df.drop('price', axis=1)
    df['price'] = moved_column
    
    ### Features selecting
    
    
    ## numerical features:
    
    ## corr matrix
    
    correlation_matrix = df.corr()
    plt.figure(figsize=(16, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
    
    # Shold I drop 'room_number'?
    
    ## pps
    
    columns = ['Area', 'floor', 'price','hasElevator', 'hasParking', 'hasBars', 'hasStorage', 'hasAirCondition', 'hasBalcony', 'hasMamad', 'handicapFriendly','num_of_images']

    feature_scores = {}
    for column in columns:
        score = pps.score(df, column, 'price')
        feature_scores[column] = score['ppscore']
    
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("Predictive Power Scores:")
    for feature, score in sorted_features:
        print(f"{feature}: {score}")

    # Shold I drop ['floor','hasElevator','hasBars','hasStorage','hasAirCondition','handicapFriendly','num_of_images'] ?
    
    ## Variance Inflation Factor
    
    numerical_columns = ['Area', 'floor','num_of_images']
    binary_columns = ['hasElevator', 'hasParking', 'hasBars', 'hasStorage', 'hasAirCondition', 'hasBalcony', 'hasMamad', 'handicapFriendly']
    
    selected_columns = numerical_columns + binary_columns
    
    subset_df = df[selected_columns]
    
    subset_df = sm.add_constant(subset_df)
    
    vif = pd.DataFrame()
    vif["Feature"] = subset_df.columns
    vif["VIF"] = [variance_inflation_factor(subset_df.values, i) for i in range(subset_df.shape[1])]
    
    print("Variance Inflation Factor (VIF):")
    print(vif)
    
    # Nothing to remove here..
    # I'll remove the columns from the PPS and the corr matrix:
    
    df = df.drop(columns=['room_number']) # from the corr matrix
    df = df.drop(columns=['hasElevator','hasBars','hasStorage','hasAirCondition','handicapFriendly','num_of_images']) #floor didn't improve the model
    
    ## Categorial features:
    
    ## Chi square test
    
    categorical_columns = ['City', 'type', 'city_area', 'condition', 'furniture', 'entrance_date','Street']

    selected_features = []
    results = pd.DataFrame(columns=['Feature', 'Chi2', 'P-value'])
    
    for column in categorical_columns:
        contingency_table = pd.crosstab(df[column], df['price'])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        results = results.append({'Feature': column, 'Chi2': chi2, 'P-value': p_value}, ignore_index=True)
        if p_value < 0.05:
            selected_features.append(column)
    
    print("Selected Categorical Features:", selected_features)
    
    print("\nChi-square Test Results:")
    print(results)
    
    # All seems good..
    
    ## Mutual Information
    
    # categorical_columns = ['City', 'type', 'city_area', 'condition', 'furniture', 'entrance_date','Street']

    # encoded_df = pd.get_dummies(df[categorical_columns], drop_first=True)
    
    # feature_scores = mutual_info_regression(encoded_df, df['price'])
    
    # results = pd.DataFrame({'Feature': encoded_df.columns, 'Mutual Information': feature_scores})
    
    # sorted_features = results.sort_values(by='Mutual Information', ascending=False)
    
    # sorted_features[['Feature_1', 'Feature_2']] = sorted_features['Feature'].str.rsplit('_', n=1, expand=True)

    # sorted_features = sorted_features.drop(columns=['Feature'])
    
    # sorted_features.groupby('Feature_1').mean()
    
    # Maybe we need to remove the 'city_area' and 'Street' features?
    
    ## Information Gain
    
    # categorical_columns = ['City', 'type', 'city_area', 'condition', 'furniture', 'entrance_date','Street']

    # encoded_df = pd.get_dummies(df[categorical_columns], drop_first=True)
    
    # feature_scores = mutual_info_classif(encoded_df, df['price'])
    
    # results = pd.DataFrame({'Feature': encoded_df.columns, 'Information Gain': feature_scores})
    
    # sorted_features = results.sort_values(by='Information Gain', ascending=False)
        
    # sorted_features[['Feature_1', 'Feature_2']] = sorted_features['Feature'].str.rsplit('_', n=1, expand=True)

    # sorted_features = sorted_features.drop(columns=['Feature'])
    
    # sorted_features.groupby('Feature_1').mean()
    
    # Also here the 'city_area' and 'Street' features got the lowest score.
    # I'll remove them:
        
    #df = df.drop(columns=['city_area']) #It didn't improve the model
    df = df.drop(columns=['Street'])
    
    #df.to_csv('df_prepared.csv', index=False, encoding='utf-8-sig')
    return df