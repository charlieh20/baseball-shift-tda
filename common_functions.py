import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import random
import persim

def histogram_poly_fit(dataframe, num_bins=18, degree=3, weighted=True):
    # Extract 'angle_from_first_base_line' and 'dist' columns from the DataFrame
    angles = dataframe['angle_from_first_base_line']
    distances = dataframe['dist']

    # Set bins for histogram
    bin_width = math.floor(90 / num_bins)
    bins = list(range(0, 91, bin_width))

    weights = None
    if weighted:
        # Min-max scale the 'dist' column to the range [0, 1]
        scaler = MinMaxScaler()
        scaled_distances = scaler.fit_transform(distances.values.reshape(-1, 1))
        weights = scaled_distances.flatten()

    # Create a histogram with specified bins
    heights, bins = np.histogram(angles, bins=bins, weights=weights)

    # Calculate bin centers from bin edges
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fit a polynomial to the data using bin centers
    coefficients = np.polyfit(bin_centers, heights, degree)
    polynomial = np.poly1d(coefficients)

    # Plot the fitted polynomial
    x_range = (bin_centers[0], bin_centers[-1])

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((heights - polynomial(bin_centers))**2)
    
    return polynomial, x_range, mse


def find_optimal_degree(df, degrees=range(5), tol=0.01):
    vals = []
    for idx, d in enumerate(degrees):
        _, _, mse = histogram_poly_fit(df, degree=d, weighted=True)
        vals.append(mse)
        if idx > 0 and abs(vals[idx] - vals[idx-1]) < tol: # want 'one more' than optimal, more fit = clearer features
            return d
    return d


def get_extrema(poly, x_range):
    # Getting interval critical points
    crit = poly.deriv().r
    r_crit = np.append(crit[crit.imag==0].real, x_range)
    sorted_crit = np.sort(r_crit)
    return sorted_crit, poly(sorted_crit)



def level_set_pdgm_points(y_crit):
    # Convert critical point y-coordinate data into birth and death points
    min_filt = [y_crit[i] < y_crit[i+1] for i in range(len(y_crit)-1)]
    min_filt.append(y_crit[-1] < y_crit[-2])
    y_min = y_crit[min_filt]
    y_max = y_crit[np.invert(min_filt)]

    if len(y_min) == len(y_max):
        y_max = y_max[:-1]
    y_max = np.append(y_max, 1.1 * max(y_crit))

    births = np.sort(y_min)
    deaths = np.sort(y_max)
    deaths = np.concatenate(([deaths[-1]], deaths[:-1]))

###

    ## fix for scale issue... divide by max value in either axis... histogram density=True didn't seem to work...
    return births[1:] / max(max(births[1:]), max(deaths[1:])), deaths[1:] / max(max(births[1:]), max(deaths[1:]))

###



def df_to_pdgm(df, degree):
    poly_line, poly_range, _ = histogram_poly_fit(df, degree=degree, weighted=True)
    _, y_crit = get_extrema(poly_line, poly_range)
    x, y = (level_set_pdgm_points(y_crit))
    
###
    ## fix for issue where len(x) != len(y)... not sure why this is happening though...
    if len(x) != len(y):
        return [[x[i], y[i]] for i in range(min(len(x), len(y)))]
    else:
        return [[x[i], y[i]] for i in range(len(x))]
###


def random_W_distance(diagram_lst1, diagram_lst2, dbg=False):
    pd1 = np.array(random.choice(diagram_lst1))
    pd2 = np.array(random.choice(diagram_lst2))
    dist = persim.sliced_wasserstein(pd1, pd2, M=100)
                    
    if dbg and dist > 100:
        print(pd1, pd2, dist)
    return dist