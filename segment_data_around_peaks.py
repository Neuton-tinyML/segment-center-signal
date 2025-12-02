import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def prepare_data(data, step, work_wind_size):
    """
    Prepares data for further analysis by creating a list of square root values
    based on sliding windows of a specified size.

    Parameters:
    -----------
    data : list
        A list of numerical values to be processed.
    step : int
        The step size between the start of each window.
    work_wind_size : int
        The size of the sliding window.

    Returns:
    --------
    list
        A list of square root values, one for each sliding window of the specified size.
    
    """
    prepared_data = []
    for i in range(0, len(data) - work_wind_size,  step):
        data_window = data[i : i + work_wind_size]
        prepared_data.append(np.sqrt( np.square(np.min(data_window)) + np.square(np.max(data_window)) ))
    return prepared_data


def create_segmented_df(df, segments, total_wind_size):
    """
    Creates a new DataFrame by segmenting the input DataFrame based on specified segments.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to be segmented.
    segments : list of tuples
        A list of tuples, where each tuple represents a segment of the input DataFrame.
        Each tuple should contain two integers, representing the start and end indices of the segment.
    total_wind_size : int
        The total size of the window to be created around the center index of each segment.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame containing the rows of the input DataFrame that fall within the specified segments,
        as well as additional rows from the input DataFrame that fall within a window of size `total_wind_size`
        centered around the midpoint of each segment.
    
    """
    index_labled = 0
    columns = []

    for column in df:
        columns.append(column)
        df[column] = df[column].astype(int)

    df_segmented = pd.DataFrame(columns=columns)
    
    for s in segments:
        start_index, end_index = s
        index_center = int((end_index + start_index) / 2)
        for i in range(index_center - int(total_wind_size / 2), index_center + int(total_wind_size / 2)):
            row_data = df.iloc[i]
            df_segmented.loc[index_labled] = row_data
            index_labled += 1
    return df_segmented

def segment_data(data, work_wind_size, threshold_coef) :
    """
    Segments the input data based on zero-crossings above a specified threshold.

    Parameters:
    -----------
    data : list or numpy.ndarray
        The input data to be segmented.
    work_wind_size : int
        The size of the sliding window to be used for segmenting the data.
    threshold_coef : float
        The coefficient to be used for calculating the threshold for zero-crossing detection.
        The threshold is calculated as `threshold_coef` times the mean of the input data.

    Returns:
    --------
    list of tuples
        A list of tuples, where each tuple represents a segment of the input data.
        Each tuple contains two integers, representing the start and end indices of the segment.
        The indices are shifted by `work_wind_size / 2` to account for the sliding window.
    
    """
    global config
    threshold_coef = config['threshold_coef']

    # Calculate the mean of the data
    mean_data = np.mean(data)
    # Set the threshold for zero-crossing detection
    threshold = threshold_coef * mean_data

    i = 0
    segment_started = False
    start_index = 0
    end_index = 0
    segments = []
    
    for d in  data:
        i += 1
        if d > threshold:
            if not segment_started:
                segment_started = True
                start_index = i
        else :
            if segment_started:
                segment_started = False
                end_index = i

                segments.append((start_index + work_wind_size / 2, end_index + work_wind_size / 2))

    return segments

def get_vector_of_work_axis(df):
    """
    Extracts a vector of values from a specified column of a pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame from which to extract the vector of values.

    Returns:
    --------
    numpy.ndarray
        A one-dimensional numpy array containing the values from the specified column of the input DataFrame.
    
    """
    global config
    work_axis = config['work_axis']
    return df[work_axis].values.reshape(( len(df[work_axis]) ))


def plot_segments(df, n_segments=20):
    """
    Plots n_segments of a specified column of a pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame from which to extract the segments.

    Returns:
    --------
    None
        The function only generates a plot of the specified segments.
    
    """    
    global config
    total_wind_size = config['total_wind_size']
    work_axis = config['work_axis']
    for i in range(n_segments):
        start_ix = i*total_wind_size
        end_ix = i*total_wind_size + total_wind_size
        sample = df.loc[start_ix : end_ix]
        sample.plot(title=f'Start ix: {start_ix}, End ix: {end_ix}')


def main(df):
    global config
    step = config['step']
    work_wind_size = config['work_wind_size']
    total_wind_size = config['total_wind_size']
    threshold_coef = config['threshold_coef']

    axis_vector = get_vector_of_work_axis(df)
    # Substract DC offset
    dc = np.mean(axis_vector)
    data = axis_vector - dc
    data = prepare_data(data, step, work_wind_size)
    segments = segment_data(data, work_wind_size, threshold_coef)
    df_segmented = create_segmented_df(df, segments, total_wind_size)
    return df_segmented


# =======================================================================
# MAIN SCRIPT EXECUTION EXAMPLE

# Define the desired training window size
TRAINING_WINDOW_SIZE = 100
gesture = 'swipe_right'

# read data with one non-continuous gesture samples
df = pd.read_csv('sample_data.csv', on_bad_lines='skip')
# make sure there are no NaN values
assert df.isnull().sum().sum() == 0
# set proper column names
df.columns = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ', 'target']
# reset index
df.reset_index(drop=True, inplace=True)


config = {
    'work_axis' : 'aY', # tunable parameter
    'work_wind_size': int(TRAINING_WINDOW_SIZE * 0.95), # tunable parameter
    'total_wind_size': TRAINING_WINDOW_SIZE,
    'threshold_coef': 0.5, # tunable parameter
    'step': 1
    }

# plot first 5 windows before segmentation (for comparison)
plot_segments(df, n_segments = 5)

# segment data around detected peaks
df_segmented = main(df)

# plot first 5 windows after segmentation (for comparison)
plot_segments(df_segmented, n_segments = 5)

# save processed data
df_segmented.to_csv(f'processed_{gesture}.csv', index = False)

