# Import
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Adds previous stock prices and scales the features
def prepare_for_binary(fin_dataframe):

    # Define a function to add price movement
    def label_price_movement(dataframe, column_name):
        dataframe[f'{column_name.lower()}_movement'] = (dataframe.groupby('cik')[column_name].diff().fillna(0) > 0).astype(int)

    # Add price movement for each price column
    price_columns = ['Open', 'High', 'Low', 'Close', 'Avg_Price']
    for column in price_columns:
        label_price_movement(fin_dataframe, column)

    # Separate basic_info, features, targets
    basic_info = fin_dataframe.iloc[:, :6]
    features = fin_dataframe.iloc[:, 6:-5]
    targets = fin_dataframe.iloc[:, -5:]

    # Create the scaler
    scaler = MinMaxScaler()

    # Fit the scaler to "feature" data and scale it
    scaled_features = scaler.fit_transform(features)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

    # Concatenate all DFs
    scaled_done_df = pd.concat([basic_info, scaled_features_df, targets], axis=1)
    return scaled_done_df