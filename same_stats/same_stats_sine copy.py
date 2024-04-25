import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import random
import pytweening
from random import randrange
import os
import json

# update lacation of results


def get_values(df):
    """Calculates the summary statistics for the given set of points

    Args:
        df (pd.DataFrame): A ``DataFrame`` with ``x`` and ``y`` columns

    Returns:
        list: ``[x-mean, y-mean, x-stdev, y-stdev, correlation]``
    """
    xm = df.quantile(.05, axis=1)
    xl = df.quantile(.95, axis=1)
    return [*xm, *xl]

def is_error_still_ok(df1, df2, decimals=2):
    """Checks to see if the statistics are still within the acceptable bounds

    Args:
        df1 (pd.DataFrame): The original data set
        df2 (pd.DataFrame): The test data set
        decimals (int):     The number of decimals of precision to check

    Returns:
        bool: ``True`` if the maximum error is acceptable, ``False`` otherwise
    """
    r1 = get_values(df1)
    r1 = [math.floor(r * 10**decimals) for r in r1]

    r2 = get_values(df2)
    r2 = [math.floor(r * 10**decimals) for r in r2]

    # we are good if r1 and r2 have the same numbers
    er = np.subtract(r1, r2)
    er = [abs(n) for n in er]
    return np.max(er) == 0

def save_png(df_old, df_new, final_parameters,  df_sin, directory, i):
    if not os.path.exists(directory):
        os.makedirs(directory)
        # Plot the final_parameters
    for column in df_new.columns:
        plt.plot(df_old.index, df_old[column], color = 'yellow', alpha=0.7)
        plt.plot(df_new.index, df_new[column], color='blue', alpha=0.7)
        #Plot the reference sine wave
        parameters = final_parameters[column]
        reference_sine = sine_function(np.arange(len(df_new)), *parameters)
        plt.plot(df_new.index, reference_sine, linestyle='dashdot',  color='green', alpha=0.7)

    # get the overall statistics for function
    plt.plot(df_sin.index, get_values(df_sin)[:100], linestyle='-', color='orange', alpha=0.7, label = "quantile created sine waves")
    # plt.plot(df_sin.index, get_values(df_sin)[100:], linestyle='-', color='orange', alpha=0.7)

    # get the overall statistics for old and new dataframe
    plt.plot(df_new.index, get_values(df_new)[:100], linestyle='-', linewidth=3, color='red', alpha=0.7, label = "quantile new data")
    plt.plot(df_old.index, get_values(df_old)[:100], linestyle='dashdot', linewidth=2, color='purple', alpha=0.7, label = "quantile old data")

    
    # plt.plot(df_new.index, get_values(df_new)[100:], linestyle='-', linewidth=3, color='red', alpha=0.7)
    # plt.plot(df_old.index, get_values(df_old)[100:], linestyle='dashdot', linewidth=2, color='purple', alpha=0.7)
    plt.title('Original and Fitted Sine Waves')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(directory, "iteration" + str(i) + "_save_5.png"))
    plt.close()

        # Plot the final_parameters
    for column in df_new.columns[:5]:
        plt.plot(df_old.index, df_old[column], color = 'yellow', alpha=0.7)
        plt.plot(df_new.index, df_new[column], color='blue', alpha=0.7)
        #Plot the reference sine wave
        parameters = final_parameters[column]
        reference_sine = sine_function(np.arange(len(df_new)), *parameters)
        plt.plot(df_new.index, reference_sine, linestyle='dashdot',  color='green', alpha=0.7)

    # get the overall statistics for function
    plt.plot(df_sin.index, get_values(df_sin)[:100], linestyle='-', color='orange', alpha=0.7, label = "quantile created sine waves")
    plt.plot(df_sin.index, get_values(df_sin)[100:], linestyle='-', color='orange', alpha=0.7)

    # get the overall statistics for old and new dataframe
    plt.plot(df_new.index, get_values(df_new)[:100], linestyle='-', linewidth=3, color='red', alpha=0.7, label = "quantile new data")
    plt.plot(df_old.index, get_values(df_old)[:100], linestyle='dashdot', linewidth=2, color='purple', alpha=0.7, label = "quantile old data")

    
    plt.plot(df_new.index, get_values(df_new)[100:], linestyle='-', linewidth=3, color='red', alpha=0.7)
    plt.plot(df_old.index, get_values(df_old)[100:], linestyle='dashdot', linewidth=2, color='purple', alpha=0.7)
    plt.title('Original and Fitted Sine Waves')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(directory, "iteration" + str(i) + "_save_all.png"))
    plt.close()
    json_object = json.dumps(final_parameters, indent = 4) 

    df_new.to_csv(os.path.join(directory, directory + "_new_df.csv"))
    df_old.to_csv(os.path.join(directory, directory + "_old_df.csv"))

    df_sin.to_csv(os.path.join(directory, directory + "_sin_df.csv"))
    return


def fit_sine_to_series(df, column, parameters_random):
    # get the column
    series = df[column]
    # get the parameters
    parameters_random_column = parameters_random[column]
    # Generate x values starting from 0
    x_data = np.arange(len(series))
    
    # create new function
    sine_function_created = create_function_offset(*parameters_random_column)

    # get the parameter of the fitted function (which is jsut the vertical offset)
    parameters, _ = curve_fit(sine_function_created, x_data, series)

    # get how good the fit is with the function
    fitted_sine = sine_function_created(x_data, *parameters)
    current_deviation = np.mean(np.abs(series - fitted_sine))

    # get all the parameters - the randomly generated variables, and the fitted offset
    new_param = list(parameters_random_column)
    new_param.extend(list(parameters))
    return new_param, current_deviation

def s_curve(v):
    return pytweening.easeInOutQuad(v)

def fit_sine_to_dataframe(df_old, directory,  max_jitter=.1, num_iterations=3000):
    final_parameters = {}
    sum_difference = {}
    parameters_random = {}
    df_new = df_old.copy()
    df_sin = df_old.copy()
    for column in df_old:
        # create the random parameters for the function
        sum_difference[column] = np.inf
        parameters = [np.random.random_sample(), 0.5*np.random.random_sample(), 10*np.random.random_sample()]
        parameters_random[column] = parameters
    
    for j in range(num_iterations):
        for column in df_new.columns:
            print(j)
            # Generate random jitter for y
            t = (0.1-.01) * s_curve(((num_iterations - j) / num_iterations)) + .01
            y = 0
            while True:
                y +=1
                df_test = df_new.copy()
                row = np.random.randint(0, df_old.shape[0])
                jitter_y = np.random.uniform(-max_jitter, max_jitter)
                df_test.loc[row, column] =  df_new.loc[row, column]  + jitter_y
                parameters, new_column_difference = fit_sine_to_series(df_test, column, parameters_random)

                # if y%2 ==0: 
                #     column2 = randrange(len(df_new.columns))
                #     df_test.loc[row, column2] =  df_new.loc[row, column2]  - jitter_y
                #     parameters2, new_column_difference2 = fit_sine_to_series(df_test, column2, parameters_random)
                #     old_diff = sum_difference[column] + sum_difference[column2]
                #     new_diff = new_column_difference + new_column_difference2
                    
                # else:
                old_diff = sum_difference[column] 
                new_diff = new_column_difference 
                #get the reference sin function (where the lines will move to)
                if j == 0:
                    reference_sine = sine_function(np.arange(len(df_new)), *parameters)
                    df_sin[column] = reference_sine
                

                do_bad = np.random.random_sample() < t
                if new_diff < old_diff or do_bad:
                    if is_error_still_ok(df_old, df_test, decimals = 2):
                        final_parameters[column] = parameters
                        sum_difference[column] = new_column_difference
                        # if y%2 ==0: 
                        #     final_parameters[column2] = parameters2
                        #     sum_difference[column2] = new_column_difference2
                        df_new = df_test
                        break

        if j % 1000 == 0:
            save_png(df_old, df_new, final_parameters, df_sin, directory, j)
    return df_new, final_parameters


#df_old = pd.DataFrame(pd.DataFrame(np.tile(np.random.randn(100), (100,1)), index=np.arange(100)))
# use the same default starting point of lines
directory = "paper_sin_05"

def initial_function(x, b):
     return np.power(x, .25) + b

def sine_function(x, ar, br, cr, dr):
        return np.power(x, .25) + ar*np.sin(br*x + cr) + dr


def create_function_offset(ar, br, cr):
        # create new target function for each line, where only D (the horizontal setting) will change as result of location of line
        # the other parameters are set randomly
        def sine_function(x, D):
            return np.power(x, .25) + ar*np.sin(br*x + cr) + D
        return sine_function

df_old = pd.read_csv(os.path.join("paper_sin", "paper_sin_old_df.csv"))
df_sqrt = df_old.copy()
for column in df_old.columns:
        df_sqrt_column = initial_function(np.arange(len(df_old)), *[np.random.randn()])
        df_sqrt[column] = df_sqrt_column


# Fit sine wave model to each series
df_new, final_parameters = fit_sine_to_dataframe(df_sqrt, directory)


#  current_articles = [(el[0], el['finished_ner_chunk']) for el in session.query(Article).with_entities(Article.gkg_record_id).all()]

