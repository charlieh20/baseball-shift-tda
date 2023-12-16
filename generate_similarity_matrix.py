import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from common_functions import find_optimal_degree, df_to_pdgm, random_W_distance


def similarity_matrix_teams(df, subset_sample_percentage=0.8, number_of_pd=100, pairwise_comparisons=100):
    
    MLB_teams = list(set(df.pitch_team.values))

    dgms_by_teams = {}
    
    for team in MLB_teams:
        
        tmp_df = df[df.pitch_team == team]
        
        optimal_degree = find_optimal_degree(tmp_df)
        
        for _ in range(number_of_pd):

            tmp = tmp_df.sample(int(subset_sample_percentage * tmp_df.shape[0]))

            diagram_points = df_to_pdgm(tmp, optimal_degree)
            
            if team not in dgms_by_teams:
                dgms_by_teams[team] = []
                
            dgms_by_teams[team].append(diagram_points)
            
                   
    matrix_dictionary = {}
    visit = set()

    for i in range(len(MLB_teams)):
        team1 = MLB_teams[i]
        for j in range(i, len(MLB_teams)):
            team2 = MLB_teams[j]

            if (team1, team2) not in visit and (team2, team1) not in visit:
                
                matrix_dictionary[(team1, team2)] = []
                
                for _ in range(pairwise_comparisons):
                    
                    dist = random_W_distance(dgms_by_teams[team1], dgms_by_teams[team2])
                    
                    if dist > 100:
                        print(team1, team2)
                 
                    matrix_dictionary[(team1, team2)].append(dist)
                
                matrix_dictionary[(team1, team2)] = sum(matrix_dictionary[(team1, team2)]) / len(matrix_dictionary[(team1, team2)])
                if i != j:
                    matrix_dictionary[(team2, team1)] = matrix_dictionary[(team1, team2)]

    return matrix_dictionary

def similarity_matrix_dates(df, subset_sample_percentage=0.8, number_of_pd=100, pairwise_comparisons=100):

    #df[(pd.to_datetime(df.date).dt.year == 2023) & (pd.to_datetime(df.date).dt.month == 9)]
    filter_dates = [
        [2022, None], 
        [2023, 4], 
        [2023, 5], 
        [2023, 6], 
        [2023, 7], 
        [2023, 8], 
        [2023, 9]
        ]

    dgms_by_date = {}

    for dates in filter_dates:
        
        date_key = str(dates[0]) + str(dates[1])
        
        if not dates[1]:
            tmp_df = df[pd.to_datetime(df.date).dt.year == dates[0]]
        else:
            tmp_df = df[(pd.to_datetime(df.date).dt.year == dates[0]) & (pd.to_datetime(df.date).dt.month == dates[1])]
        
        optimal_degree = find_optimal_degree(tmp_df)
        
        for _ in range(number_of_pd):
            
            tmp = tmp_df.sample(int(subset_sample_percentage * tmp_df.shape[0]))
            
            diagram_points = df_to_pdgm(tmp, optimal_degree)
            
            if date_key not in dgms_by_date:
                dgms_by_date[date_key] = []
            dgms_by_date[date_key].append(diagram_points)
        
    matrix_dictionary = {}
    visit = set()

    for i in range(len(filter_dates)):
        date1 = filter_dates[i]
        for j in range(i, len(filter_dates)):
            date2 = filter_dates[j]

            date1_key = str(date1[0]) + str(date1[1])
            date2_key = str(date2[0]) + str(date2[1])
            
            if (date1_key, date2_key) not in visit and (date2_key, date1_key) not in visit:
                
                matrix_dictionary[(date1_key, date2_key)] = []
                
                for _ in range(pairwise_comparisons):
                    
                    dist = random_W_distance(dgms_by_date[date1_key], dgms_by_date[date2_key])
                    
                    if dist > 100:
                        print(date1_key, date2_key)
                    
                    matrix_dictionary[(date1_key, date2_key)].append(dist)
                
                matrix_dictionary[(date1_key, date2_key)] = sum(matrix_dictionary[(date1_key, date2_key)]) / len(matrix_dictionary[(date1_key, date2_key)])
                if i != j:
                    matrix_dictionary[(date2_key, date1_key)] = matrix_dictionary[(date1_key, date2_key)]

    return matrix_dictionary

def similarity_matrix_handedness(df, subset_sample_percentage=0.8, number_of_pd=100, pairwise_comparisons=100):

    #df[(pd.to_datetime(df.date).dt.year == 2023) & (pd.to_datetime(df.date).dt.month == 9)]
    filters = ["L", "R"]

    dgms_by_hand = {}

    for hand in filters:
        
        tmp_df = df[df.batter_handedness == hand]
        
        optimal_degree = find_optimal_degree(tmp_df)
        
        for _ in range(number_of_pd):
            
            tmp = tmp_df.sample(int(subset_sample_percentage * tmp_df.shape[0]))
            
            diagram_points = df_to_pdgm(tmp, optimal_degree)
            
            if hand not in dgms_by_hand:
                dgms_by_hand[hand] = []
            dgms_by_hand[hand].append(diagram_points)
        
    matrix_dictionary = {}
    visit = set()

    for hand1 in filters:
        for hand2 in filters:
            
            if (hand1, hand2) not in visit and (hand2, hand1) not in visit:
                
                matrix_dictionary[(hand1, hand2)] = []
                
                for _ in range(pairwise_comparisons):
                    
                    dist = random_W_distance(dgms_by_hand[hand1], dgms_by_hand[hand2])
                    
                    if dist > 100:
                        print(hand1, hand2)
                    
                    matrix_dictionary[(hand1, hand2)].append(dist)
                
                matrix_dictionary[(hand1, hand2)] = sum(matrix_dictionary[(hand1, hand2)]) / len(matrix_dictionary[(hand1, hand2)])

    return matrix_dictionary



def similarity_matrix_rank(df, subset_sample_percentage=0.8, number_of_pd=100, pairwise_comparisons=100):

    dgms_by_rnk = {}

    ranges = [range(i * 60 + 1, (i+1) * 60 + 1) for i in range(0, 3)]
    range_keys = [f'Ranks {i * 60 + 1} - {(i+1) * 60}' for i in range(0, 3)]

    for i in range(len(ranges)):
        rnk_range = ranges[i]
        tmp_key = range_keys[i]
        tmp_df = df[(df.shift_rank >= min(rnk_range)) & (df.shift_rank <= max(rnk_range))]
        
        optimal_degree = find_optimal_degree(tmp_df)
        
        for _ in range(number_of_pd):
            
            tmp = tmp_df.sample(int(subset_sample_percentage * tmp_df.shape[0]))
            
            diagram_points = df_to_pdgm(tmp, optimal_degree)
            
            if tmp_key not in dgms_by_rnk:
                dgms_by_rnk[tmp_key] = []
            dgms_by_rnk[tmp_key].append(diagram_points)
        
    matrix_dictionary = {}
    
    visit = set()

    for i in range(len(range_keys)):
        rnk1 = range_keys[i]
        for j in range(i, len(range_keys)):
            rnk2 = range_keys[j]
            if (rnk1, rnk2) not in visit and (rnk2, rnk1) not in visit:
                
                matrix_dictionary[(rnk1, rnk2)] = []
                
                for _ in range(pairwise_comparisons):
                    
                    dist = random_W_distance(dgms_by_rnk[rnk1], dgms_by_rnk[rnk2])
                    
                    if dist > 100:
                        print(rnk1, rnk2)
                    
                    matrix_dictionary[(rnk1, rnk2)].append(dist)
                
                matrix_dictionary[(rnk1, rnk2)] = sum(matrix_dictionary[(rnk1, rnk2)]) / len(matrix_dictionary[(rnk1, rnk2)])
                if i != j:
                    matrix_dictionary[(rnk2, rnk1)] = matrix_dictionary[(rnk1, rnk2)]

    return matrix_dictionary


def similarity_matrix_dates_year(df, subset_sample_percentage=0.8, number_of_pd=100, pairwise_comparisons=100):

    #df[(pd.to_datetime(df.date).dt.year == 2023) & (pd.to_datetime(df.date).dt.month == 9)]
    filter_dates = [2022, 2023] 

    dgms_by_date = {}

    for dates in filter_dates:
        
        date_key = str(dates)
        
        tmp_df = df[pd.to_datetime(df.date).dt.year == dates]
        
        optimal_degree = find_optimal_degree(tmp_df)
        
        for _ in range(number_of_pd):
            
            tmp = tmp_df.sample(int(subset_sample_percentage * tmp_df.shape[0]))
            
            diagram_points = df_to_pdgm(tmp, optimal_degree)
            
            if date_key not in dgms_by_date:
                dgms_by_date[date_key] = []
            dgms_by_date[date_key].append(diagram_points)
        
    matrix_dictionary = {}
    visit = set()

    for date1 in filter_dates:
        for date2 in filter_dates:
            date1_key = str(date1)
            date2_key = str(date2)
            
            if (date1_key, date2_key) not in visit and (date2_key, date1_key) not in visit:
                
                matrix_dictionary[(date1_key, date2_key)] = []
                
                for _ in range(pairwise_comparisons):
                    
                    dist = random_W_distance(dgms_by_date[date1_key], dgms_by_date[date2_key])
                    
                    if dist > 100:
                        print(date1_key, date2_key)
                    
                    matrix_dictionary[(date1_key, date2_key)].append(dist)
                
                matrix_dictionary[(date1_key, date2_key)] = sum(matrix_dictionary[(date1_key, date2_key)]) / len(matrix_dictionary[(date1_key, date2_key)])

    return matrix_dictionary



def similarity_matrix_teams_dates(df, subset_sample_percentage=0.8, number_of_pd=100, pairwise_comparisons=100):

    #df[(pd.to_datetime(df.date).dt.year == 2023) & (pd.to_datetime(df.date).dt.month == 9)]

    dgms_by_date = {}
    
    MLB_teams = list(set(df.pitch_team.values))
    years = [2022, 2023]
    unique_keys = set()

    for yr in years:
        
        for team in MLB_teams:
        
            date_key = team + str(yr)
            
            unique_keys.add(date_key)
            
            tmp_df = df[(pd.to_datetime(df.date).dt.year == yr) & (df.pitch_team == team)]
            
            optimal_degree = find_optimal_degree(tmp_df)
            
            for _ in range(number_of_pd):
                
                tmp = tmp_df.sample(int(subset_sample_percentage * tmp_df.shape[0]))
                
                try:
                    diagram_points = df_to_pdgm(tmp, optimal_degree)
                
                except:
                    _ -= 1
                    continue
                
                if date_key not in dgms_by_date:
                    dgms_by_date[date_key] = []
                dgms_by_date[date_key].append(diagram_points)

    matrix_dictionary = {}
    visit = set()

    unique_keys = sorted(list(unique_keys))

    for i in range(len(unique_keys)):
        key1 = unique_keys[i]
        for j in range(i, len(unique_keys)):
            key2 = unique_keys[j]
            
            if (key1, key2) not in visit and (key2, key1) not in visit and key1 in dgms_by_date and key2 in dgms_by_date:
                
                matrix_dictionary[(key1, key2)] = []
                
                for _ in range(pairwise_comparisons):
                    
                    dist = random_W_distance(dgms_by_date[key1], dgms_by_date[key2])
                    
                    if dist > 100:
                        print(key1, key2)
                    
                    matrix_dictionary[(key1, key2)].append(dist)
                
                matrix_dictionary[(key1, key2)] = sum(matrix_dictionary[(key1, key2)]) / len(matrix_dictionary[(key1, key2)])
                if i != j:
                    matrix_dictionary[(key2, key1)] = matrix_dictionary[(key1, key2)]

    return matrix_dictionary

def similarity_matrix_years_hand(df, subset_sample_percentage=0.8, number_of_pd=100, pairwise_comparisons=100):

    dgms_by_date = {}
    
    years = [2022, 2023]
    hands = ["L", "R"]

    unique_keys = set()

    for yr in years:
        
        for hand in hands:
        
            date_key =  f'{str(yr)}, {hand}'
            
            unique_keys.add(date_key)
            
            tmp_df = df[(pd.to_datetime(df.date).dt.year == yr) & (df.batter_handedness == hand)]
            
            optimal_degree = find_optimal_degree(tmp_df)
            
            for _ in range(number_of_pd):
                
                tmp = tmp_df.sample(int(subset_sample_percentage * tmp_df.shape[0]))
                
                try:
                    diagram_points = df_to_pdgm(tmp, optimal_degree)
                
                except:
                    _ -= 1
                    continue
                
                if date_key not in dgms_by_date:
                    dgms_by_date[date_key] = []
                dgms_by_date[date_key].append(diagram_points)

    matrix_dictionary = {}
    visit = set()

    unique_keys = sorted(list(unique_keys))

    for i in range(len(unique_keys)):
        key1 = unique_keys[i]
        for j in range(i, len(unique_keys)):
            key2 = unique_keys[j]
            
            if (key1, key2) not in visit and (key2, key1) not in visit and key1 in dgms_by_date and key2 in dgms_by_date:
                
                matrix_dictionary[(key1, key2)] = []
                
                for _ in range(pairwise_comparisons):
                    
                    dist = random_W_distance(dgms_by_date[key1], dgms_by_date[key2])
                    
                    if dist > 100:
                        print(key1, key2)
                    
                    matrix_dictionary[(key1, key2)].append(dist)
                
                matrix_dictionary[(key1, key2)] = sum(matrix_dictionary[(key1, key2)]) / len(matrix_dictionary[(key1, key2)])
                if i != j:
                    matrix_dictionary[(key2, key1)] = matrix_dictionary[(key1, key2)]

    return matrix_dictionary


def plot_sim_matrix(scores, figsize, group_by, sort=True, values=None):
    # Extract unique x and y values
    if values:
        x_values = values
        y_values = values
    else:
        x_values = list(set(key[0] for key in scores.keys()))
        y_values = list(set(key[1] for key in scores.keys()))
        if sort:
            x_values = sorted(x_values)
            y_values = sorted(y_values)

    # Create a matrix using NumPy
    matrix = np.zeros((len(y_values), len(x_values)))

    # Fill in the matrix with data from the dictionary
    for key, value in scores.items():
        x_index = x_values.index(key[0])
        y_index = y_values.index(key[1])
        matrix[y_index, x_index] = value

    # Increase figure size
    plt.figure(figsize=figsize)

    # Create the plot with annotations
    im = plt.imshow(matrix, interpolation='nearest', cmap='viridis', origin='lower')

    # Add values in each cell
    for i in range(len(y_values)):
        for j in range(len(x_values)):
            plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='w')

    # Set tick labels
    plt.xticks(np.arange(len(x_values)), x_values)
    plt.yticks(np.arange(len(y_values)), y_values)

    plt.title(f'Similarity Matrix, Dgms Grouped by {group_by}')

    # Show the plot
    plt.show()