# Let's define some functions we'll use in the notebook
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from scipy.stats import ks_2samp, chi2_contingency, shapiro, f_oneway, levene, chi2_contingency
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import  mutual_info_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif




def compare_dataframes(df1, df2, df1_name="DataFrame 1", df2_name="DataFrame 2"):
    """
    Compares two DataFrames to check if the number and names of columns are the same.
    
    Args:
    df1 (pd.DataFrame): The first DataFrame to compare.
    df2 (pd.DataFrame): The second DataFrame to compare.
    df1_name (str): The name of the first DataFrame (for display purposes).
    df2_name (str): The name of the second DataFrame (for display purposes).
    
    Returns:
    None
    """
    columns_df1 = df1.columns
    columns_df2 = df2.columns

    if len(columns_df1) != len(columns_df2):
        print(f"The number of columns in {df1_name} and {df2_name} are different.")
        print(f"{df1_name} has {len(columns_df1)} columns, while {df2_name} has {len(columns_df2)} columns.")
    else:
        print(f"The number of columns in {df1_name} and {df2_name} are the same.")
    
    if set(columns_df1) != set(columns_df2):
        print(f"The column names in {df1_name} and {df2_name} are different.")
        print(f"{df1_name} columns: {list(columns_df1)}")
        print(f"{df2_name} columns: {list(columns_df2)}")
    else:
        print(f"The column names in {df1_name} and {df2_name} are the same.")
        

        
def kolmogorov_smirnov_test(X_train, X_test, alpha=0.05):
    """
    Perform the Kolmogorov-Smirnov test on each column of two DataFrames.
    
    Args:
    X_train (pd.DataFrame): The first DataFrame.
    X_test (pd.DataFrame): The second DataFrame.
    alpha (float): The significance level. Default is 0.05.
    
    Returns:
    None
    """
    p_values = []  # Initialize an empty list to store p-values
    all_not_rejected = True  # Flag to track if we don't reject the null hypothesis for all columns
    
    # Iterate over each column in X_train
    for col in X_train.columns:
        # Perform the KS test on the corresponding columns from X_train and X_test
        statistic, p_value = ks_2samp(X_train[col], X_test[col])
        p_values.append(p_value)  # Append the p-value to the list
        
        # If any p-value is less than or equal to alpha, set the flag to False
        if p_value <= alpha:
            all_not_rejected = False
    
    # Check if the null hypothesis is not rejected for all columns
    if all_not_rejected:
        print("Not enough evidence to reject the null hypothesis for all columns.")
    else:
        # Iterate over each column and its corresponding p-value
        for col, p_value in zip(X_train.columns, p_values):
            # If p-value is less than or equal to alpha, print that the null hypothesis is rejected for the column
            if p_value <= alpha:
                print(f"Reject the null hypothesis for column {col}, the two samples do not come from the same distribution.")
        


def is_series_equal_to_range(series, start, end):
    """
    Checks if a pandas Series is equal to a range of values.
    
    Args:
    series (pd.Series): The pandas Series to check.
    start (int): The starting value of the range (inclusive).
    end (int): The ending value of the range (exclusive).
    
    Returns:
    bool: True if the Series is equal to the range of values, False otherwise.
    """
    # Create a range of values
    range_values = range(start, end)
    
    # Check if the Series has the same length as the range
    if len(series) != len(range_values):
        return False
    
    # Compare each value of the Series with the corresponding value of the range
    return all(series.iloc[i] == range_values[i] for i in range(len(range_values)))


    
def check_duplicate_rows(df: pd.DataFrame) -> Optional[bool]:
    """
    Check for duplicate rows in a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to check for duplicates.

    Returns:
    Optional[bool]: Returns True if duplicates are found, False if no duplicates are found, 
                    and None if the DataFrame is empty.
    """
    
    # Check if the DataFrame is empty
    if df.empty:
        logging.info("The DataFrame is empty.")  # Log an info message if DataFrame is empty
        return None  # Return None if DataFrame is empty

    # Identify duplicate rows in the DataFrame
    duplicate_rows = df[df.duplicated()]

    # Check if there are any duplicate rows
    if not duplicate_rows.empty:
        logging.warning("There are duplicate rows in the DataFrame.")  # Log a warning message if duplicates are found

        # Print the duplicate rows
        print("There are duplicate rows in the dataframe:")
        print(duplicate_rows)
        return True  # Return True indicating that duplicates are found
    else:
        # Print message if no duplicate rows are found
        print("There are NO duplicate rows in the dataframe.")
        return False  # Return False indicating no duplicates
 

    
def find_missing_columns(df: pd.DataFrame) -> None:
    """
    Identify columns in a DataFrame that have missing values and print the number of missing values and their proportion 
    relative to the total number of rows for each column.

    Args:
    df (pd.DataFrame): The DataFrame to check for missing values.

    Returns:
    None
    """
    
    # Identify columns with missing values
    missing_info = df.isnull().sum()
    
    # Filter columns that have missing values
    missing_columns = missing_info[missing_info > 0]
    
    
    if not missing_columns.empty:
        print("Columns with missing values, their counts, and their proportions:")
        for col, missing_count in missing_columns.items():
            missing_proportion = missing_count / len(df)
            print(f"{col}: {missing_count} missing values ({missing_proportion:.2%} of total rows)")
    else:
        print("There are no columns with missing values.")
        

        
def fill_na_with_regression(row, B0, B1):
    """
    Function to maintain the original value if it's not NaN.
    If it's NaN, fill it with the regression prediction.
    
    Args:
    row (pd.Series): The row of the DataFrame.
    B0 (float): Intercept of the linear regression.
    B1 (float): Slope of the linear regression.
    
    Returns:
    float: Original value or predicted value.
    """
    if not np.isnan(row["Arrival Delay in Minutes"]):
        return row["Arrival Delay in Minutes"]
    else:
        return B0 + B1 * row["Departure Delay in Minutes"]
    

    
def count_nan_rows(df, df_name):
    """
    Count the number of rows in a DataFrame with at least one NaN value and calculate the proportion.

    Args:
    df (pd.DataFrame): The DataFrame to check.

    Returns:
    tuple: A tuple containing the count of rows with NaN values and their proportion.
    """
    nan_rows_count = df.isna().any(axis=1).sum()
    nan_rows_proportion = nan_rows_count / len(df)
    print(f"Number of rows in {df_name} with at least one NaN value:", round(nan_rows_count,2))
    print(f"Proportion of rows in {df_name} with at least one NaN value relative to the total number of rows: {nan_rows_proportion:.2%}")

    
    
def shapiro_wilk_test(X_train, alpha=0.05):
    """
    Function to perform the Shapiro-Wilk test for normality for each column in the dataframe.
    
    Parameters:
    X_train (pd.DataFrame): A dataframe containing the columns to be tested.
    alpha (float): Significance level to determine if the null hypothesis can be rejected. Default is 0.05.
    
    Returns:
    None: Prints the result of the Shapiro-Wilk test for each column.
    """
    # Initialize a list to store p-values
    p_values = []
    
    # Loop through each column in the dataframe
    for col in X_train.columns:
        # Perform Shapiro-Wilk test
        statistic, p_value = shapiro(X_train[col])
        # Append the p-value to the list
        p_values.append(p_value)
    
    # Loop through each column and its corresponding p-value
    for col, p_value in zip(X_train.columns, p_values):
        # Check if the p-value is greater than the significance level alpha
        if p_value > alpha:
            # If p > alpha, we do not reject the null hypothesis (normality assumption is met)
            print(f"Not enough evidence to reject the null hypothesis for {col}.")
        else:
            # If p <= alpha, we reject the null hypothesis (normality assumption is violated)
            print(f"Reject the null hypothesis for column {col}. The variable is not normally distributed.") 
            
            

def anova_correlation(dataframe_numeric, dataframe_categorical, alpha=0.05):
    """
    Function to perform ANOVA (Analysis of Variance) to test the difference in means for each numeric column in the 
    dataframe against a categorical series.
    
    Parameters:
    dataframe_numeric (pd.DataFrame): A dataframe containing the numeric columns to be tested.
    dataframe_categorical (pd.Series): A series containing the categorical variable to test against.
    alpha (float): Significance level to determine if the null hypothesis can be rejected. Default is 0.05.
    
    Returns:
    None: Prints the result of the ANOVA test for each column.
    """
    # Loop through each numeric column in the dataframe
    for column in dataframe_numeric.columns:
        # Perform one-way ANOVA test
        f_statistic, p_value = f_oneway(dataframe_numeric[column], dataframe_categorical)
        
        # Check if the p-value is greater than the significance level alpha
        if p_value > alpha:
            # If p > alpha, we do not reject the null hypothesis
            print(f"No sufficient evidence to reject the null hypothesis for {column}.")
        else:
            # If p <= alpha, we reject the null hypothesis
            print(f"Reject the null hypothesis for column {column}. There is a significant difference in means.")          
                        
                
                
def check_homoscedasticity(dataframe_numeric, dataframe_categorical, alpha=0.05):
    """
    Function to perform Levene's test for homoscedasticity (equal variances) for each numeric column in the dataframe
    against a categorical series.
    
    Parameters:
    dataframe_numeric (pd.DataFrame): A dataframe containing the numeric columns to be tested.
    dataframe_categorical (pd.Series): A series containing the categorical variable to test against.
    alpha (float): Significance level to determine if the null hypothesis can be rejected. Default is 0.05.
    
    Returns:
    None: Prints the result of the homoscedasticity test for each column.
    """
    # Get unique categories from the categorical series
    categories = dataframe_categorical.unique()
    
    # Loop through each numeric column in the dataframe
    for column in dataframe_numeric.columns:
        # Group data by categories
        grouped_data = [dataframe_numeric[column][dataframe_categorical == category] for category in categories]
        # Perform Levene's test on the grouped data
        statistic, p_value = levene(*grouped_data)
        
        # Check if the p-value is greater than the significance level alpha
        if p_value > alpha:
            # If p > alpha, we do not reject the null hypothesis (homoscedasticity is met)
            print(f"Not enough evidence to reject the null hypothesis for {column}. Homoscedasticity assumption is met.")
        else:
            # If p <= alpha, we reject the null hypothesis (homoscedasticity is violated)
            print(f"Reject the null hypothesis for {column}. Homoscedasticity assumption is violated.")

            
                        
def chi_square_test(dataframe, series, alpha=0.05):
    """
    Function to perform chi-square test of independence for each column in the dataframe against a categorical series.
    
    Parameters:
    dataframe (pd.DataFrame): A dataframe containing the columns to be tested.
    series (pd.Series): A series containing the categorical variable to test against.
    alpha (float): Significance level to determine if the null hypothesis can be rejected. Default is 0.05.
    
    Returns:
    None: Prints the result of the hypothesis test for each column.
    """
    for column in dataframe.columns:
        # Create a contingency table between the column and the series
        contingency_table = pd.crosstab(dataframe[column], series)
        # Perform the chi-square test on the contingency table
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        
        # Check if the p-value is greater than the significance level alpha
        if p > alpha:
            # If p > alpha, we do not reject the null hypothesis
            print(f"Not enough evidence to reject the null hypothesis for {column}. The variables are independent.")
        else:
            # If p <= alpha, we reject the null hypothesis
            print(f"Reject the null hypothesis for {column}. The variables are not independent.")
            
            
            
def chi_square_test_two_dataframes(df1, df2, series):
    # Inizializza una lista per memorizzare i risultati
    results = []

    # Itera su ogni colonna dei dataframe
    for col in df1.columns:
        # Crea una tabella di contingenza per il primo dataframe
        contingency_table_1 = pd.crosstab(df1[col], series)
        chi2_1, p_1, dof_1, ex_1 = chi2_contingency(contingency_table_1)
        
        # Crea una tabella di contingenza per il secondo dataframe
        contingency_table_2 = pd.crosstab(df2[col], series)
        chi2_2, p_2, dof_2, ex_2 = chi2_contingency(contingency_table_2)
        
        # Aggiungi i risultati alla lista
        results.append({'column': col, 'chi2_df1': chi2_1, 'chi2_df2': chi2_2, 'p_df1': p_1, 'p_df2': p_2})
    
    # Converti i risultati in un dataframe
    results_df = pd.DataFrame(results)
    
    return results_df
            
            
            
def is_correlated_chisquared(data):
    """
    Calculate p-values for chi-squared test between all pairs of columns in the dataset.

    Args:
        data (DataFrame): Dataset containing columns for analysis.

    Returns:
        DataFrame: DataFrame containing p-values for chi-squared test between all pairs of columns.
    """
    # Get all the columns of the dataset
    columns = data.columns
    
    # Initialize a matrix to store p-values
    p_values_matrix = np.zeros((len(columns), len(columns)))
    
    # Loop over all pairs of columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            
            # Calculate the contingency table for the pair of columns
            ct = pd.crosstab(index=data[col1], columns=data[col2])
            
            # Perform the chi-squared test
            chi_sq_result = chi2_contingency(ct)
            
            # Extract the p-value
            p = chi_sq_result[1]
            
            # Store the p-value in the matrix
            p_values_matrix[i, j] = p
            p_values_matrix[j, i] = p
    
    # Create a DataFrame from the matrix
    p_values_df = pd.DataFrame(p_values_matrix, index=columns, columns=columns)
    
    return p_values_df.round(2)



def is_correlated_mutualinfo(data):
    """
    Calculate mutual information between all pairs of columns in the dataset.

    Args:
        data (DataFrame): Dataset containing columns for analysis.

    Returns:
        DataFrame: DataFrame containing mutual information scores between all pairs of columns.
    """
    # Get all the columns of the dataset
    columns = data.columns
    
    # Initialize a matrix to store mutual information scores
    mi_matrix = np.zeros((len(columns), len(columns)))
    
    # Loop over all pairs of columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            
            # Calculate mutual information for the pair of columns
            mi = mutual_info_score(data[col1], data[col2])
            
            # Store the mutual information score in the matrix
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    
    # Create a DataFrame from the matrix
    mi_df = pd.DataFrame(mi_matrix, index=columns, columns=columns)
    
    return mi_df.round(2)
   
    
    
def plot_correlation_matrix(p_values_df):
    """
    Plot a heatmap of p-values obtained from chi-squared test.

    Args:
        p_values_df (DataFrame): DataFrame containing p-values.

    Returns:
        None
    """
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))  # Set the size of the figure

    # Create a heatmap
    sns.heatmap(p_values_df, annot=True, cmap='coolwarm', center=0.5, 
                cbar_kws={'label': 'p-value'}, vmin=0, vmax=1)

    # Add titles and labels
    plt.title('Chi-Squared Test p-value Matrix')  # Set the title of the plot
    plt.xlabel('Columns')  # Label for the x-axis
    plt.ylabel('Columns')  # Label for the y-axis

    # Show the plot
    plt.show()  # Display the heatmap
    
    
    
def select_k_best_features(score_func, train, target_column, k=5):
    """
    Function to perform feature selection using SelectKBest.
    
    Parameters:
    score_func (callable): Scoring function to use (chi2, mutual_info_classif, or f_classif).
    train (pd.DataFrame): Training dataframe with features and target.
    target_column (str): Name of the target column.
    k (int): Number of top features to select. Default is 5.
    
    Returns:
    None
    """
    # Define the feature selector with the given scoring function and number of features to select
    selector = SelectKBest(score_func, k=k)
    
    # Separate the features and target from the training dataframe
    X = train.drop([target_column], axis=1)
    y = train[target_column]
    
    # Fit the selector to the data and transform it
    X_new = selector.fit_transform(X, y)
    
    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)
    
    # Get the names of the selected features
    selected_feature_names = X.columns[selected_indices]
    
    # Get the name of the scoring function
    method_name = score_func.__name__
    
    # Print the selected feature names with k and method
    print(f"Selected {k} features using method {method_name}: {list(selected_feature_names)}")

    
    
def define_models(models=dict()):
    """
    This function defines and returns a dictionary of ensemble models from the sklearn library.

    Parameters:
    models (dict): A dictionary to which new models will be added. Defaults to an empty dictionary if not provided.

    Returns:
    dict: A dictionary containing the defined ensemble models.
    """

    # Add an AdaBoostClassifier to the models dictionary
    models['ada'] = AdaBoostClassifier(random_state=42)

    # Add a RandomForestClassifier to the models dictionary
    models['rf'] = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Add an ExtraTreesClassifier to the models dictionary
    models['et'] = ExtraTreesClassifier(random_state=42, n_jobs=-1)

    print('Defined %d models' % len(models))

    return models



def make_pipeline(model):
    """
    This function creates and returns a machine learning pipeline that includes preprocessing steps and a specified model.

    Parameters:
    model: The machine learning model to be included in the pipeline.

    Returns:
    Pipeline: A sklearn pipeline object that preprocesses the data and then applies the model.
    """

    # Define the columns for transformations
    label_encoding_cols = ["Gender", "Customer Type", "Type of Travel"]  
    ordinal_encoding_cols = ["Class"]  

    # Order for the OrdinalEncoder
    order = [["Eco", "Eco Plus", "Business"]]

    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='if_binary'), label_encoding_cols),  # Apply OneHotEncoder to binary columns
            ('ordinal', OrdinalEncoder(categories=order), ordinal_encoding_cols)  # Apply OrdinalEncoder to ordered columns
        ],
        remainder='passthrough'  # Keep the other columns unchanged
    )

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Add the preprocessor to the pipeline
        ('model', model)  # Add the model to the pipeline
    ])

    return pipeline  # Return the complete pipeline



def make_pipeline2(model):
    """
    This function creates and returns a machine learning pipeline that includes preprocessing steps and a specified model.

    Parameters:
    model: The machine learning model to be included in the pipeline.

    Returns:
    Pipeline: A sklearn pipeline object that preprocesses the data and then applies the model.
    """

    # Define the columns for transformations
    label_encoding_cols = ["Type of Travel"]  
    ordinal_encoding_cols = ["Class"]  

    # Order for the OrdinalEncoder
    order = [["Eco", "Eco Plus", "Business"]]

    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='if_binary'), label_encoding_cols),  # Apply OneHotEncoder to binary columns
            ('ordinal', OrdinalEncoder(categories=order), ordinal_encoding_cols)  # Apply OrdinalEncoder to ordered columns
        ],
        remainder='passthrough'  # Keep the other columns unchanged
    )

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Add the preprocessor to the pipeline
        ('model', model)  # Add the model to the pipeline
    ])

    return pipeline  # Return the complete pipeline



def evaluate_model(X, y, model, folds=5, metric="accuracy"):
    """
    This function evaluates a machine learning model using cross-validation.

    Parameters:
    X: Features data (input variables).
    y: Target data (output variable).
    model: The machine learning model to be evaluated.
    folds (int): The number of cross-validation folds. Default is 5.
    metric (str): The evaluation metric to be used. Default is "accuracy".

    Returns:
    array: Array of scores from cross-validation.
    """

    # Create the pipeline
    pipeline = make_pipeline(model)
    # Create a machine learning pipeline that includes preprocessing steps and the specified model.

    # Evaluate the model
    scores = cross_val_score(pipeline, X, y, scoring=metric, cv=folds, n_jobs=-1)
    # Perform cross-validation and return the evaluation scores.
    # n_jobs=-1 utilizes all available CPU cores for parallel processing.

    return scores
    # Return the array of cross-validation scores.



def evaluate_model2(X, y, model, folds=5, metric="accuracy"):
    """
    This function evaluates a machine learning model using cross-validation.

    Parameters:
    X: Features data (input variables).
    y: Target data (output variable).
    model: The machine learning model to be evaluated.
    folds (int): The number of cross-validation folds. Default is 5.
    metric (str): The evaluation metric to be used. Default is "accuracy".

    Returns:
    array: Array of scores from cross-validation.
    """

    # Create the pipeline
    pipeline = make_pipeline2(model)
    # Create a machine learning pipeline that includes preprocessing steps and the specified model.

    # Evaluate the model
    scores = cross_val_score(pipeline, X, y, scoring=metric, cv=folds, n_jobs=-1)
    # Perform cross-validation and return the evaluation scores.
    # n_jobs=-1 utilizes all available CPU cores for parallel processing.

    return scores
    # Return the array of cross-validation scores.

    
    
def robust_evaluate_model(X, y, model, folds=5, metric="accuracy"):
    """
    This function robustly evaluates a machine learning model using cross-validation, suppressing warnings and handling exceptions.

    Parameters:
    X: Features data (input variables).
    y: Target data (output variable).
    model: The machine learning model to be evaluated.
    folds (int): The number of cross-validation folds. Default is 5.
    metric (str): The evaluation metric to be used. Default is "accuracy".

    Returns:
    array or None: Array of scores from cross-validation if successful, None if an exception occurs.
    """

    scores = None  # Initialize scores to None

    try:
        # Suppress warnings during model evaluation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")  # Ignore all warnings
            scores = evaluate_model(X, y, model, folds, metric)  # Evaluate the model
    except:
        # If an exception occurs, set scores to None
        scores = None

    return scores  # Return the evaluation scores or None if an exception occurred



def robust_evaluate_model2(X, y, model, folds=5, metric="accuracy"):
    """
    This function robustly evaluates a machine learning model using cross-validation, suppressing warnings and handling exceptions.

    Parameters:
    X: Features data (input variables).
    y: Target data (output variable).
    model: The machine learning model to be evaluated.
    folds (int): The number of cross-validation folds. Default is 5.
    metric (str): The evaluation metric to be used. Default is "accuracy".

    Returns:
    array or None: Array of scores from cross-validation if successful, None if an exception occurs.
    """

    scores = None  # Initialize scores to None

    try:
        # Suppress warnings during model evaluation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")  # Ignore all warnings
            scores = evaluate_model2(X, y, model, folds, metric)  # Evaluate the model
    except:
        # If an exception occurs, set scores to None
        scores = None

    return scores  # Return the evaluation scores or None if an exception occurred



def evaluate_models(X, y, models, folds=5, metric="accuracy"):
    """
    This function evaluates multiple machine learning models using cross-validation, 
    handling exceptions and suppressing warnings.

    Parameters:
    X: Features data (input variables).
    y: Target data (output variable).
    models (dict): A dictionary of machine learning models to be evaluated.
    folds (int): The number of cross-validation folds. Default is 5.
    metric (str): The evaluation metric to be used. Default is "accuracy".

    Returns:
    dict: A dictionary containing the evaluation scores for each model.
    """

    results = dict()  # Initialize an empty dictionary to store results

    for name, model in models.items():
        # Evaluate the model
        scores = robust_evaluate_model(X, y, model, folds, metric)

        # Show process
        if scores is not None:
            # Store the result
            results[name] = scores
            mean_score, std_score = np.mean(scores), np.std(scores)
            print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))  # Print the mean and standard deviation of the scores
        else:
            print('>%s: error' % name)  # Print an error message if the model evaluation failed

    return results  # Return the dictionary of results



def evaluate_models2(X, y, models, folds=5, metric="accuracy"):
    """
    This function evaluates multiple machine learning models using cross-validation, 
    handling exceptions and suppressing warnings.

    Parameters:
    X: Features data (input variables).
    y: Target data (output variable).
    models (dict): A dictionary of machine learning models to be evaluated.
    folds (int): The number of cross-validation folds. Default is 5.
    metric (str): The evaluation metric to be used. Default is "accuracy".

    Returns:
    dict: A dictionary containing the evaluation scores for each model.
    """

    results = dict()  # Initialize an empty dictionary to store results

    for name, model in models.items():
        # Evaluate the model
        scores = robust_evaluate_model2(X, y, model, folds, metric)

        # Show process
        if scores is not None:
            # Store the result
            results[name] = scores
            mean_score, std_score = np.mean(scores), np.std(scores)
            print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))  # Print the mean and standard deviation of the scores
        else:
            print('>%s: error' % name)  # Print an error message if the model evaluation failed

    return results  # Return the dictionary of results



def summarize_results(results, maximize=True, top_n=2):
    """
    This function summarizes the results of model evaluations by printing the top N models
    and displaying a boxplot of their scores.

    Parameters:
    results (dict): A dictionary containing the evaluation scores for each model.
    maximize (bool): Whether to maximize the evaluation metric (e.g., accuracy). Default is True.
    top_n (int): The number of top models to summarize. Default is 2.

    Returns:
    None
    """

    # Check for no results
    if len(results) == 0:
        print('no results')
        return

    # Determine how many results to summarize
    n = min(top_n, len(results))

    # Create a list of (name, mean(scores)) tuples
    mean_scores = [(k, np.mean(v)) for k, v in results.items()]

    # Sort tuples by mean score
    mean_scores = sorted(mean_scores, key=lambda x: x[1])

    # Reverse for descending order (e.g., for accuracy)
    if maximize:
        mean_scores = list(reversed(mean_scores))

    # Retrieve the top n for summarization
    names = [x[0] for x in mean_scores[:n]]
    scores = [results[x[0]] for x in mean_scores[:n]]

    # Print the top n
    print()
    for i in range(n):
        name = names[i]
        mean_score, std_score = np.mean(results[name]), np.std(results[name])
        print('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i + 1, name, mean_score, std_score))

    # Boxplot for the top n
    plt.boxplot(scores, labels=names)
    _, labels = plt.xticks()
    plt.setp(labels, rotation=90)



def plot_chi_squared_selection_features(data, target_column):
    """
    Computes chi-squared statistics and p-values for features in the given dataset,
    converts the chi-squared statistics to a pandas Series, sorts the values, and plots a bar chart.

    Parameters:
    data (pd.DataFrame): The input dataset containing features and the target column.
    target_column (str): The name of the target column in the dataset.

    Returns:
    None: This function displays a bar plot of the sorted chi-squared statistics.
    """
    # Drop the target column from the dataset to get feature columns
    feature_columns = data.drop([target_column], axis=1)
    
    # Compute chi-squared statistics and p-values
    chi2_stats, p_values = chi2(feature_columns, data[target_column])
    
    # Convert chi-squared statistics to a pandas Series
    chi2_series = pd.Series(chi2_stats, index=feature_columns.columns)
    
    # Sort the chi-squared statistics in descending order
    chi2_series_sorted = chi2_series.sort_values(ascending=False)
    
    # Plot the sorted chi-squared statistics as a bar chart
    plt.figure(figsize=(20, 8))
    chi2_series_sorted.plot.bar()
    plt.title('Chi-Squared Features Selection')
    plt.xlabel('Features')
    plt.ylabel('Chi-Squared Statistic')
    plt.show()



def plot_mutual_information_selection_features(data, target_column):
    """
    Computes mutual information scores for features in the given dataset,
    converts the mutual information scores to a pandas Series, sorts the values, and plots a bar chart.

    Parameters:
    data (pd.DataFrame): The input dataset containing features and the target column.
    target_column (str): The name of the target column in the dataset.

    Returns:
    None: This function displays a bar plot of the sorted mutual information scores.
    """
    # Drop the target column from the dataset to get feature columns
    feature_columns = data.drop([target_column], axis=1)
    
    # Compute mutual information scores
    mutual_info = mutual_info_classif(feature_columns, data[target_column])
    
    # Convert mutual information scores to a pandas Series
    mutual_info_series = pd.Series(mutual_info, index=feature_columns.columns)
    
    # Sort the mutual information scores in descending order
    mutual_info_series_sorted = mutual_info_series.sort_values(ascending=False)
    
    # Plot the sorted mutual information scores as a bar chart
    plt.figure(figsize=(20, 8))
    mutual_info_series_sorted.plot.bar()
    plt.title('Mutual Information Features Selection')
    plt.xlabel('Features')
    plt.ylabel('Mutual Information Score')
    plt.show()



def plot_f_values_selection_features(data, target_column):
    """
    Computes F-values for features in the given dataset,
    converts the F-values to a pandas Series, sorts the values, and plots a bar chart.

    Parameters:
    data (pd.DataFrame): The input dataset containing features and the target column.
    target_column (str): The name of the target column in the dataset.

    Returns:
    None: This function displays a bar plot of the sorted F-values.
    """
    # Drop the target column from the dataset to get feature columns
    feature_columns = data.drop([target_column], axis=1)
    
    # Compute F-values and p-values
    f_values, p_values = f_classif(feature_columns, data[target_column])
    
    # Convert F-values to a pandas Series
    f_values_series = pd.Series(f_values, index=feature_columns.columns)
    
    # Sort the F-values in descending order
    f_values_series_sorted = f_values_series.sort_values(ascending=False)
    
    # Plot the sorted F-values as a bar chart
    plt.figure(figsize=(20, 8))
    f_values_series_sorted.plot.bar()
    plt.title('F-values Features Selection')
    plt.xlabel('Features')
    plt.ylabel('F-value')
    plt.show()



def hyper_tuning(X, y, classifier_type, list_label):
    order = ["Eco", "Eco Plus", "Business"]

    # Preprocessing transformer
    transformer = ColumnTransformer(
        transformers=[
            ("categorical_dicotomic_values", OneHotEncoder(drop="if_binary"), list_label),
            ("categorical_multiple_values", OrdinalEncoder(categories=[order]), ["Class"])
        ],
        remainder='passthrough'
    )

    # Choose classifier based on input
    if classifier_type == 'RandomForest':
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif classifier_type == 'ExtraTrees':
        classifier = ExtraTreesClassifier(random_state=42, n_jobs=-1)
    else:
        raise ValueError("Invalid classifier_type. Choose 'RandomForest' or 'ExtraTrees'.")

    # Pipeline for preprocessing and classifier
    pipeline = Pipeline([
        ('transformer', transformer),
        ('classifier', classifier)
    ])

    # Definition of the parameter grid
    param_grid = {
        'classifier__n_estimators': [100, 150, 200],
        'classifier__criterion': ["gini", "entropy", "log_loss"],
        'classifier__min_samples_split': [2, 3, 4],
        'classifier__min_samples_leaf': [1, 2, 3],
        'classifier__max_features': ["sqrt", "log2"]
    }

    # Configure GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

    # Fit the grid search to your data
    grid_search.fit(X, y)

    # Print the best parameters found
    print("Best parameters found: ", grid_search.best_params_)

    # Extract cross-validation scores
    best_index = grid_search.best_index_
    best_score = grid_search.cv_results_['mean_test_score'][best_index]

    # Return the best estimator and the cross-validation score
    return grid_search.best_estimator_, best_score