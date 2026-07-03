def add_features(df):
    df['TotalSF'] = (df['TotalBsmtSF'] +
                     df['1stFlrSF'] +
                     df['2ndFlrSF'])

    df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    df['Remodeled'] = (
        df['YearRemodAdd'] != df['YearBuilt']
    ).astype(int)

    df['TotalBaths'] = (df['FullBath'] +
                        0.5 * df['HalfBath'] +
                        df['BsmtFullBath'] +
                        0.5 * df['BsmtHalfBath'])

    df['LuxuryScore'] = df['OverallQual'] * df['OverallCond']

    return df