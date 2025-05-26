import pandas as pd


def concat_reduced_chunks(df1, df2):

    basic_info = df1.iloc[:, :6]
    features = df1.iloc[:, 6:-72]
    prev_prices_targets = df1.iloc[:, -72:]

    basic_info_2 = df2.iloc[:, :6]
    features_2 = df2.iloc[:, 6:-72]
    prev_prices_targets_2 = df2.iloc[:, -72:]

    col_2 = list(df2.columns)
    columns_to_drop = []

    for column in col_2:
        if column in features.columns:
            columns_to_drop.append(column)

    features_df2_cropped = features_2.drop(columns=columns_to_drop)

    concat_basic_info_df1_df2 = pd.concat([basic_info, basic_info_2])
    concat_feature_df1_df2 = pd.concat([features, features_df2_cropped], axis=1)
    concat_prev_prices_targets_df1_df2 = pd.concat([prev_prices_targets, prev_prices_targets_2])
    final_df = pd.concat([concat_basic_info_df1_df2, concat_feature_df1_df2, concat_prev_prices_targets_df1_df2], axis=1)
    
    final_df.fillna(0, inplace=True)

    return final_df