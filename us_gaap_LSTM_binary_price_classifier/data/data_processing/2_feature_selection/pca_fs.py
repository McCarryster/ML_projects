from sklearn.decomposition import PCA
import pandas as pd

def select_with_pca(features, threshold, with_name=True):

    # Create a PCA object
    pca = PCA()

    # This is where we do all the PCA math (calculate loading scores and the variation each PC accounts for)
    pca.fit(features)

    # Get the explained variance ratio of each component
    explained_variance = pca.explained_variance_ratio_

    # Find the number of components that explain most of the variance
    cumulative_variance = 0
    num_components = 0
    for i, variance in enumerate(explained_variance):
        cumulative_variance += variance
        if cumulative_variance >= threshold:
            num_components = i + 1
            break
    print('num_components =', num_components)

    # Fit PCA with the selected number of components
    pca = PCA(n_components=num_components)
    pca.fit(features)

    if with_name == True:
        # Get the indices of the most valuable columns
        most_valuable_columns = [features.columns[i] for i in range(num_components)]
        # Drop all columns except the most valuable ones
        features_reduced = features[most_valuable_columns]
        return features_reduced
    else:
        # Get the transformed data with reduced dimensions
        df_pca = pd.DataFrame(pca.transform(features))
        return df_pca

