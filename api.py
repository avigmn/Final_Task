from flask import Flask, request, render_template
import pickle
import os
import pandas as pd
import re
import numpy as np
from datetime import datetime, timedelta


app = Flask(__name__)
my_model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.getlist('feature')
    
    
    column_names = ['City', 'type', 'Area', 'city_area', 'hasParking', 'condition', 'hasBalcony', 'hasMamad', 'furniture', 'floor', 'entrance_date']
    df = pd.DataFrame([features], columns=column_names)
    print(df.info())
    
    try:
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
        
        df["type"] = df["type"].replace("קוטג'", "קוטג")
        df["type"] = df["type"].replace("קוטג' טורי", "קוטג טורי")
        
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
        
        df['city_area'] = df['city_area'].apply(clean_name)
        df['city_area'] = df['city_area'].astype(str)
        df['city_area'] = df['city_area'].replace('nan', '')
        
        df["floor"] = df["floor"].replace('קומת קרקע' , 0)
        df["floor"] = df["floor"].replace('קומת מרתף' , -1)
        df["floor"] = pd.to_numeric(df["floor"], errors="coerce")
        
        df.loc[(df['type'] == 'בית פרטי') & (~df['floor'].isin([0, np.nan])), 'floor'] = 0
        df.loc[(df['type'] == 'קוטג') & (~df['floor'].isin([0, np.nan])), 'floor'] = 0
        df.loc[(df['type'] == 'קוטג טורי') & (~df['floor'].isin([0, np.nan])), 'floor'] = 0
        df.loc[(df['type'] == 'דו משפחתי') & (~df['floor'].isin([0, np.nan])), 'floor'] = 0
        
        df['hasParking'] = df['hasParking'].astype(str)
        df["hasParking"] = df["hasParking"].replace(["True","יש חנייה","יש","יש חניה", "yes", "כן"], 1)
        df["hasParking"] = df["hasParking"].replace(["False","אין","אין חניה","אין חנייה", "no","לא"], 0)
        df["hasParking"] = pd.to_numeric(df["hasParking"], errors="coerce")
        
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
        
        df['condition'] = df['condition'].astype(str)
        df["condition"] = df["condition"].replace(["לא צויין","nan", "None","False"], 'not_defind')
        df['condition'] = df['condition'].replace({'משופץ': 'renovated' , 'שמור': 'maintained' , 'חדש': 'new' , 'ישן': 'old' , 'דורש שיפוץ': 'requires_renovation'})
    
        df["furniture"] = df["furniture"].replace("לא צויין", "not_defined")
        df["furniture"] = df["furniture"].replace("חלקי", "partial")
        df["furniture"] = df["furniture"].replace("אין", "no")
        df["furniture"] = df["furniture"].replace("מלא", "full")
        
        df["entrance_date"] = df["entrance_date"].replace("מיידי", "less_than_6 months")
        df["entrance_date"] = df["entrance_date"].replace("גמיש", "flexible")
        df["entrance_date"] = df["entrance_date"].replace("גמיש ", "flexible")
        df["entrance_date"] = df["entrance_date"].replace("לא צויין", "not_defined")
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
        df["entrance_date"] = df["entrance_date"].apply(categorize_date)
        
    
    
        
        # df = pd.DataFrame(final_features, columns=column_names)
        # str_columns = ["City", "type", "city_area", "condition", "furniture", "entrance_date"]
        # df[str_columns] = df[str_columns].astype(str)
        
        # # Convert float columns to float type
        # float_columns = ['Area', 'floor']
        # df[float_columns] = df[float_columns].astype(float)
        
        # # Convert binary columns to int type (True will be 1, False will be 0)
        # binary_columns = ['hasParking', 'hasBalcony', 'hasMamad']
        # df[binary_columns] = df[binary_columns].astype(int)
        # print(df.columns) 
        output = my_model.predict(df)
    except:
        output = "Error - Please enter values again"            
    return render_template('index.html', prediction_text='{}'.format(output))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port,debug=True)
