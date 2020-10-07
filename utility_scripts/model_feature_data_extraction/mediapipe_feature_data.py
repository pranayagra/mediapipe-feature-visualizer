import argparse

from feature_selection import select_features

def mediapipe_feature_data(features_filepath, features, drop_na: bool = True):

    """Processes raw features extracted from MediaPipe, and
    selects the specified features for visualization.

    Parameters
    ----------
    features_filepath : str
        File path of raw mediapipe feature data to be processed

    features : list of str
        The features to extract

    drop_na : bool
        Whether or not to drop rows from the dataframe that contain NaN values

    Returns
    -------
    df : pd.DataFrame
        Selected features from mediapipe
    """

    features_no_interpolate_df = select_features(features_filepath, features,
                                  center_on_face=False, scale=1, drop_na = drop_na, do_interpolate = False)

    print(f"Select Features (Interpolation: False) DataFrame: ")
    print(features_no_interpolate_df)
    
    return features_no_interpolate_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_filepath')
    parser.add_argument('--features', default=[])
    args = parser.parse_args()

    mediapipe_feature_data(args.features_filepath, args.features)