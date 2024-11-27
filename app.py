# app.py
from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

def analyze_stocks(df):
    # Remove whitespaces in the 'NAME' column if needed
    if 'Name' in df.columns:
        df['Name'] = df['Name'].str.replace(' ', '', regex=True)

    # Select only numeric columns for filling missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Feature selection
    features = ['Sales Growth %', 'Profit Var 5Yrs %', 'Profit Var 3Yrs %', 'ROE %', 
                'P/E', 'Ind PE', 'Debt / Eq', 'CMP Rs.', 'Current ratio', 'Div Yld %', 
                'EPS 12M Rs.', 'ROA 12M %', 'CMP / Sales', 'ROCE %']
    
    features = [f for f in features if f in df.columns]
    
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    
    results = {}
    
    # Long-term potential
    if 'ROE %' in df.columns:
        y_long_term = df['ROE %'] + df['Sales Growth %']
        X_train, X_test, y_train, y_test = train_test_split(df_scaled, y_long_term, test_size=0.3, random_state=42)
        rf_regressor_long_term = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor_long_term.fit(X_train, y_train)
        df['Long_Term_Potential'] = rf_regressor_long_term.predict(df_scaled)
        results['long_term'] = df[['Name', 'Long_Term_Potential', 'CMP Rs.', 'ROE %', 'P/E']].sort_values(
            by='Long_Term_Potential', ascending=False).head(5).round(2).to_dict('records')
    
    # Short-term potential
    if 'Sales Growth %' in df.columns:
        y_short_term = df['Sales Growth %'] + df['Profit Var 3Yrs %']
        X_train, X_test, y_train, y_test = train_test_split(df_scaled, y_short_term, test_size=0.3, random_state=42)
        rf_regressor_short_term = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor_short_term.fit(X_train, y_train)
        df['Short_Term_Potential'] = rf_regressor_short_term.predict(df_scaled)
        results['short_term'] = df[['Name', 'Short_Term_Potential', 'CMP Rs.', 'ROE %', 'P/E']].sort_values(
            by='Short_Term_Potential', ascending=False).head(5).round(2).to_dict('records')
    
    # Dividend yield
    if 'Div Yld %' in df.columns:
        y_dividend = df['Div Yld %']
        X_train, X_test, y_train, y_test = train_test_split(df_scaled, y_dividend, test_size=0.3, random_state=42)
        rf_regressor_dividend = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor_dividend.fit(X_train, y_train)
        df['Dividend_Potential'] = rf_regressor_dividend.predict(df_scaled)
        results['dividend'] = df[['Name', 'Dividend_Potential', 'CMP Rs.', 'ROE %', 'P/E']].sort_values(
            by='Dividend_Potential', ascending=False).head(5).round(2).to_dict('records')
    
    # Greater returns
    if 'EPS 12M Rs.' in df.columns:
        y_returns = df['EPS 12M Rs.']
        X_train, X_test, y_train, y_test = train_test_split(df_scaled, y_returns, test_size=0.3, random_state=42)
        rf_regressor_returns = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor_returns.fit(X_train, y_train)
        df['Return_Potential'] = rf_regressor_returns.predict(df_scaled)
        results['returns'] = df[['Name', 'Return_Potential', 'CMP Rs.', 'ROE %', 'P/E']].sort_values(
            by='Return_Potential', ascending=False).head(5).round(2).to_dict('records')
    
    # Best performing
    if 'ROCE %' in df.columns:
        y_best_performing = df['ROCE %']
        X_train, X_test, y_train, y_test = train_test_split(df_scaled, y_best_performing, test_size=0.3, random_state=42)
        rf_regressor_best_performing = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor_best_performing.fit(X_train, y_train)
        df['Performance_Potential'] = rf_regressor_best_performing.predict(df_scaled)
        results['best_performing'] = df[['Name', 'Performance_Potential', 'CMP Rs.', 'ROE %', 'P/E']].sort_values(
            by='Performance_Potential', ascending=False).head(5).round(2).to_dict('records')
    
    # Penny stocks
    penny_stock_threshold = 150
    penny_stocks = df[df['CMP Rs.'] < penny_stock_threshold].copy()
    top_penny_stocks = penny_stocks.sort_values(
        by=['ROE %', 'Sales Growth %', 'P/E'], 
        ascending=[False, False, True]
    ).head(10)
    results['penny_stocks'] = top_penny_stocks[['Name', 'CMP Rs.', 'ROE %', 'Sales Growth %', 'P/E']].round(2).to_dict('records')
    
    return results

@app.route('/')
def index():
    # Load the dataset
    df = pd.read_excel('fnl_stockds_filled.xlsx')
    results = analyze_stocks(df)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)