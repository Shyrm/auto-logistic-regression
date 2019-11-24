Logistic regression wrapper
=====

Project introduces a class of logistic regression with extended functionaligy namely:  

1. Process (replace NA, test validness and convert to float) continuous variables  
2. Process (replace NA, test validness and encode) categorical variables
3. Validate model on new data using 'f1_score' and 'logloss' metrics
4. Tune model parameters following grid search cross-validation procedure  

All standard methods and parameters of 'sklearn' logistic regression are
kept: fit, predict, predict_proba. For more details on this please see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html  

**Additional parameters:**

1. categorical_variables: list of strings, default: None  
List (or other iterable collection) of the variables in data that should be considered as categorical. All other variables by default will be treated as continuous
2. continuous_na_policy: str, {'ignore', 'mean_replace', 'median_replace', 'na_mark_feature'}, default: 'median_replace'  
This parameter defines what to do with missing values in all continuous variables:  
    - 'ignore': leave missing values in variable as np.nan  
    - 'mean_replace': calculate mean value of the variable based on known values and replace missing with mean  
    - 'median_replace': calculate median value of the variable based on known values and replace missing with median  
    - 'na_mark_feature': all missing values will be replaced with 0 and additional feature 'variable_name_Unknown' will be populated. Additional feature will contain 1s where original variable had NA and 0 elsewhere.  
     
     **Notice that under 'ignore' policy all rows in data that contain at least one nan value will be excluded both from train and prediction phases!**
3. categorical_na_policy : str, 'ignore', 'new_class', default: 'new_class'  
This parameter defines what to do with missing values in all categorical variables:  
    - 'ignore': leave missing values in variable as np.nan  
    - 'new_class': treat all missing values as separate category (class) of the variable
    
     **Notice that under 'ignore' policy all rows in data that contain at least one nan value will be excluded both from train and prediction phases!**  
4. encoding_policy : str, 'ohe', 'target_encoding', default: 'ohe'  
Parameter specifies the way to encode all categorical variables
    - 'ohe': one-hot encoding (http://contrib.scikit-learn.org/categorical-encoding/onehot.html)  
    - 'target_encoding': target encoding (http://contrib.scikit-learn.org/categorical-encoding/targetencoder.html)  
5. new_category_policy : 'str', {'ignore', 'pass'}, default: 'pass'  
Parameter specifies what to do with new category in all categorical variables  
    - 'ignore': replace new category with np.nan  
    - 'pass': allow new category.  
    
    If 'pass' policy is chosen then depending on 'encoding_policy' new category will appear as follows  
    1) for 'ohe' all dummy features of known categories will take 0s
    2) for 'target_encoding' global target average will be assigned for new category  
    
    **Notice that under 'ignore' policy all rows in data that contain at least one nan value will be excluded from prediction phase!**
6. smoothing : float, default: 1.0  
Smoothing effect to balance categorical average vs prior.  
Applicable only for 'target_encoding' policy in categorical variables  
For more details check http://contrib.scikit-learn.org/categorical-encoding/targetencoder.html  
7. min_samples_leaf : int, default: 1  
Minimum samples to take category average into account  
Applicable only for 'target_encoding' policy in categorical variables  
For more details check http://contrib.scikit-learn.org/categorical-encoding/targetencoder.html  
8. special_na : list (or other iterable collection) or None, default: None  
Parameter specifies a set of special symbols that should be treated as NA  
9. special_policies : dict, default: None  
This parameter specifies special policies that should be applied for specific variables.  
Specific policies will overwrite default policies for particular variable  
Example: {'variable1': {'encoding_policy': 'target_encoding'}, 'variable2': {'encoding_policy': 'target_encoding', 'new_category_policy': 'ignore'}, ...}  


**Additional methods**

1. evaluate(X, y, threshold=0.5)  
    
    Method evaluates model performance on provided data using:  
    - f1 score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html  
    - logloss; https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html  
    
    X: dataframe to create predictions for  
    
    y: true labels associated with given data  
    
    threshold: for f1 score the probability threshold to consider class as positive. Should be in (0, 1) interval  
    
    return: dictionary of the form:{'f1_score': f1 score for given data, 'logloss': logistic loss for given data}  
2. tune_parameters(self, X, y, parameters, scoring=metrics.log_loss, greater_is_better=False, needs_proba=True, cv=3, full_results=False)  
    
    Method performs parameters tuning following grid search cross-validation strategy  

    X: features dataframe to use for parameters tuning  

    y: target labels associated with X  
    
    parameters: dictionary that defines a parameters grid to performa search over  
    
    Example: {param1: (v1, v2, ...), param2: (v1, v2, ...), ...}. For more details check https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html  
    
    scoring: function object used for evaluating model performance. Can be sklearn metric or other custom
    function that takes true labels and predictions and returns difference measure  
    
    greater_is_better: whether higher values of provided 'scoring' function imply better model performance  
    
    needs_proba: whether provided 'scoring' function needs probabilities (otherwise classes) for calculations  
    
    cv: number of cross-validation folds to use in parameters tuning  
    
    full_results: whether to return full GridSearchCV statistics. If False then only set of best parameters along with best score will be returned  
    
    return: pandas dataframe with full search statistics if full_results=True, else dictionary with best patameters combination + best cv score

**Example**

    # read data from file
    data = pd.read_csv('./Data/DR_Demo_Lending_Club_reduced.csv', sep=',', header=0)

    # drop poor features
    data.drop(['Id', 'collections_12_mths_ex_med', 'pymnt_plan', 'initial_list_status'], axis=1, inplace=True)

    # move target into separate variable
    y = data['is_bad']
    data.drop('is_bad', axis=1, inplace=True)

    # define a list of categorical variables
    cat_variables = {'addr_state',
                     'home_ownership',
                     'zip_code',
                     'policy_code',
                     'verification_status',
                     'purpose_cat'}

    # init model
    model = LogisticRegressionWrapper(
        solver='lbfgs',
        random_state=42,
        encoding_policy='target_encoding',
        categorical_na_policy='new_class',
        new_category_policy='pass',
        continuous_na_policy='median_replace',
        categorical_variables=cat_variables,
        special_na=('na',)
    )

    # fit model using all data
    model.fit(data, y)

    # print model coefficients and performance on train
    print(model.coef_)
    print(model.evaluate(data, y, threshold=np.mean(y)))

    # perform grid search for best parameters
    params_grid = {
        'encoding_policy': ('ohe', 'target_encoding'),
        'continuous_na_policy': ('median_replace', 'na_mark_feature'),
    }

    print(model.tune_parameters(data, y, params_grid))
    
**Project structure**

1. logistic_regression.py: implementation of main LogisticRegressionWrapper class  
2. categorical_variable.py: implementation of CategoricalVariable class  
3. continuous_variable.py: implementation of ContinuousVariable class  
4. test_logistic_regression.py: tests related to LogisticRegressionWrapper class  
5. test_variables.py: tests related to CategoricalVariable and ContinuousVariable classes  

**TODO list**

1. Add tests for all exceptions that ca be raised in model or variable e.g. not fitted, invalid policy, invalid values etc.  
2. Add more encoding options into categorical variable  
3. Add some transformations to continuous variable e.g. log, box-cox etc.
4. Add joint NA replacement i.e. replace one feature values based on values of one or more other features  
5. Add other than grid search parameters tuning strategy 