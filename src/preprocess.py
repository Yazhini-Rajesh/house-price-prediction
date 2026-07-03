import pandas as pd
import numpy as np

def load_and_clean(path):
    df = pd.read_csv(path)
    
    # Fix missing numbers with median
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Fix missing text with 'None'
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna('None')
    
    # Log-transform the price
    if 'SalePrice' in df.columns:
        df['SalePrice'] = np.log1p(df['SalePrice'])
    
    return df