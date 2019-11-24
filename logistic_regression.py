import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn import metrics
from continuous_variable import ContinuousVariable
from categorical_variable import CategoricalVariable
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.exceptions import NotFittedError
import warnings


class LogisticRegressionWrapper(BaseEstimator):
    """
    Class wraps 'sklearn' logistic regression model with the following functionality:

    1. Process (replace NA, test validness and convert to float) continuous variables
    For more details check 'ContinuousVariable' in 'continuous_variable.py'

    2. Process (replace NA, test validness and encode) categorical variables
    For more details check 'CategoricalVariable' in 'categorical_variable.py'

    3. Fit logistic regression model with the same parameters as in 'sklearn'
    For more details check https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    4. Get predictions from model (both probabilities and classes)

    5. Validate model on new data using 'f1_score' and 'logloss' metrics

    6. Tune model parameters following grid search cross-validation procedure

    Note that only binary classification problem is supported!

    Parameters
    ----------
    penalty : str, 'l1' or 'l2', default: 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

    dual : bool, default: False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.

    tol : float, default: 1e-4
        Tolerance for stopping criteria.

    C : float, default: 1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger regularization.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

    class_weight : dict or 'balanced', default: None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

    random_state : int, RandomState instance or None, optional, default: None
        The seed of the pseudo random number generator to use when shuffling

    solver : str, {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, default: 'lbfgs'.
        Algorithm to use in the optimization problem.

    max_iter : int, default: 100
        Useful only for the newton-cg, sag and lbfgs solvers.
        Maximum number of iterations taken for the solvers to converge.

    verbose : int, default: 0
        For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.

    warm_start : bool, default: False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    categorical_variables : list of strings, default: None
        List (or other iterable collection) of the variables in data that should be considered as categorical
        All other variables by default will be treated as continuous

    continuous_na_policy : str, {'ignore', 'mean_replace', 'median_replace', 'na_mark_feature'}, default: 'median_replace'
        This parameter defines what to do with missing values in all continuous variables:
        - 'ignore': leave missing values in variable as np.nan
        - 'mean_replace': calculate mean value of the variable based on known values and replace missing with mean
        - 'median_replace': calculate median value of the variable based on known values and replace missing with median
        - 'na_mark_feature': all missing values will be replaced with 0 and additional feature
            'variable_name_Unknown' will be populated. Additional feature will contain 1s where original variable
            had NA and 0 elsewhere.

        Notice that under 'ignore' policy all rows in data that contain at least one nan value will be excluded both
        from train and prediction phases!

    categorical_na_policy : str, 'ignore', 'new_class', default: 'new_class'
        This parameter defines what to do with missing values in all categorical variables:
        - 'ignore': leave missing values in variable as np.nan
        - 'new_class': treat all missing values as separate category (class) of the variable

        Notice that under 'ignore' policy all rows in data that contain at least one nan value will be excluded both
        from train and prediction phases!

    encoding_policy : str, 'ohe', 'target_encoding', default: 'ohe'
        Parameter specifies the way to encode all categorical variables
        - 'ohe': one-hot encoding (http://contrib.scikit-learn.org/categorical-encoding/onehot.html)
        - 'target_encoding': target encoding (http://contrib.scikit-learn.org/categorical-encoding/targetencoder.html)

    new_category_policy : 'str', {'ignore', 'pass'}, default: 'pass'
        Parameter specifies what to do with new category in all categorical variables
        - 'ignore': replace new category with np.nan
        - 'pass': allow new category.
        If 'pass' policy is chosen then depending on 'encoding_policy' new category will appear as follows
        1) for 'ohe' all dummy features of known categories will take 0s
        2) for 'target_encoding' global target average will be assigned for new category

        Notice that under 'ignore' policy all rows in data that contain at least one nan value will be excluded
        from prediction phase!

    smoothing : float, default: 1.0
        Smoothing effect to balance categorical average vs prior.
        Applicable only for 'target_encoding' policy in categorical variables
        For more details check http://contrib.scikit-learn.org/categorical-encoding/targetencoder.html

    min_samples_leaf : int, default: 1
        Minimum samples to take category average into account
        Applicable only for 'target_encoding' policy in categorical variables
        For more details check http://contrib.scikit-learn.org/categorical-encoding/targetencoder.html

    special_na : list (or other iterable collection) or None, default: None
        Parameter specifies a set of special symbols that should be treated as NA

    special_policies : dict, default: None
        This parameter specifies special policies that should be applied for specific variables.
        Specific policies will overwrite default policies for particular variable
        Example:
        {
            'variable1': {'encoding_policy': 'target_encoding'},
            'variable2': {'encoding_policy': 'target_encoding', 'new_category_policy': 'ignore'},
            ...
        }

    Attributes
    ----------
    features : dictionary of the form {variable_name -> variable object}
        In this dictionary all columns appeared in data during training phase will be stored along with associated
    objects of ContinuousVariable or CategoricalVariable.

    model : 'sklearn' logistic regression object

    Properties 'coef_' and 'intercept_' allows direct access to model.coef_ and model.intercept_ attributes. For more
    details check https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    Example
    ----------
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

    """

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1,
                 class_weight=None, random_state=None, solver='lbfgs', max_iter=100,
                 verbose=0, warm_start=False, categorical_variables=None, continuous_na_policy='median_replace',
                 categorical_na_policy='new_class', encoding_policy='ohe', new_category_policy='pass',
                 smoothing=0.5, min_samples_leaf=1, special_na=None, special_policies=None):

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose
        self.warm_start = warm_start

        self.encoding_policy = encoding_policy
        self.categorical_na_policy = categorical_na_policy
        self.new_category_policy = new_category_policy
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing

        self.continuous_na_policy = continuous_na_policy

        self.special_na = special_na

        self.special_policies = special_policies if special_policies is not None else dict()
        self.categorical_variables = categorical_variables if categorical_variables is not None else {}
        self.features = dict()
        self.model = LogisticRegression(
            penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight, random_state=random_state,
            solver=solver, max_iter=max_iter, verbose=verbose, warm_start=warm_start
        )

    @property
    def coef_(self):
        return self.model.coef_

    @property
    def intercept_(self):
        return self.model.intercept_

    def fit(self, X, y, sample_weight=None):
        """
        Fit logistic regression model using provided data
        Full procedure is comprised of the following steps:
            1. Check whether X is a pandas dataframe and y is a valid binary feature
            2. For every column in X init corresponding continuous or categorical variable object with specified policies
            3. Fit every variable object separately and store in self.features dict
            4. Create ready for modelling dataset as a result of all features 'transform' method
            5. Fit underlying logistic regression model ignoring all rows with missing values (may appear in some
                features under 'ignore' policies)
        :param X: pandas dataframe of features to fit model with
        :param y: target labels. Must be an array or series comprised of {0, 1}
        :param sample_weight: Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight
        :return: self
        """

        # check input type
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f'Expected pandas DataFrame as input but {type(X)} is given')

        # check target consistency
        if len(set(y)) != 2 or len(set(y).intersection({0, 1})) != 2:
            raise ValueError('Invalid target provided: expected binary variable in {0, 1}')

        dt = dict()
        for col in X.columns:

            if col in self.categorical_variables:

                # init variable parameters from default policies
                categorical_params = {
                    'encoding_policy': self.encoding_policy,
                    'new_category_policy': self.new_category_policy,
                    'na_policy': self.categorical_na_policy,
                    'min_samples_leaf': self.min_samples_leaf,
                    'smoothing': self.smoothing,
                    'special_na': self.special_na
                }

                # overwrite parameters with values of specific policies if available
                if col not in self.special_policies.keys():
                    params = categorical_params
                else:
                    params = {**categorical_params, **self.special_policies[col]}

                # init variable object
                var = CategoricalVariable(variable_name=col, **params)

            else:

                # init variable parameters from default policies
                continuous_params = {
                    'na_policy': self.continuous_na_policy,
                    'special_na': self.special_na
                }

                # overwrite parameters with values of specific policies if available
                if col not in self.special_policies.keys():
                    params = continuous_params
                else:
                    params = {**continuous_params, **self.special_policies[col]}

                # init variable object
                var = ContinuousVariable(variable_name=col, **params)

            var = var.fit(X[col], y)  # fit variable object
            dt = {**dt, **var.transform(X[col])}  # add transformed variable values into resulting data
            self.features[col] = var  # store variable for further usage

        # format features as pandas dataframe
        dt = pd.DataFrame(dt)

        # define valid rows mask: some features may contain NA under 'ignore' policy
        full_data_mask = ~np.any(pd.isna(dt).values, axis=1)

        # check whether there is no data left under 'ignore' policy
        if not np.any(full_data_mask):
            raise NotFittedError("Attempt to fit model with empty dataframe. This can happen under 'ignore' "
                                 "policy in some features")

        self.model.fit(dt[full_data_mask], y[full_data_mask], sample_weight)

    def _transform(self, X):
        """
        Method processes given dataframe according to specified features policies used during fit:
            1. Check whether model was fitted before
            2. For every feature stored during 'fit' phase apply values transform (or raise an error if not in data)
            3. Format all transformed values as pandas dataframe
        :param X: input data to transform as pandas dataframe
        :return: transformed features under specified policies as pandas dataframe and a logical array that marks
            all rows with at least one missing value
        """

        # check whether model is already fitted
        if not self.features:
            raise NotFittedError('Attempt to predict before "fit"')

        # check input type
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f'Expected pandas DataFrame as input but {type(X)} is given')

        dt = dict()
        for feature, feature_fact in self.features.items():

            if feature not in X.columns:
                raise KeyError(f'Missing "{feature}" column in provided dataset')

            dt = {**dt, **feature_fact.transform(X[feature])}

        # format features as pandas dataframe
        dt = pd.DataFrame(dt)

        # define valid rows mask: some features may contain NA under 'ignore' policy
        empty_data_mask = np.any(pd.isna(dt).values, axis=1)

        return dt, empty_data_mask

    def _apply(self, X, func):
        """
        Method applies func to given data taking missing values into account (will be replaced with np.nan in result)
        :param X: dataframe to apply function to
        :param func: function object to apply
        :return: results of function application
        """

        # get processed data and mask for rows to ignore
        processed_data, empty_data_mask = self._transform(X)

        # replace all values in rows to ignore with 0s
        processed_data[empty_data_mask] = 0

        # get model predictions
        res = func(processed_data)

        # replace all predictions for invalid rows with NA
        res = res.astype(np.float64)
        res[empty_data_mask] = np.nan

        return res

    def predict_proba(self, X):
        """
        Method returns predicted probabilities for given data
        :param X: dataframe to create predictions for
        :return: array with probabilities of negative and positive classes for given data
        """

        return self._apply(X, self.model.predict_proba)

    def predict(self, X):
        """
        Method returns predicted classes for given data
        :param X: datframe to create predictions for
        :return: array with classes {0, 1} for given data
        """

        return self._apply(X, self.model.predict)

    def evaluate(self, X, y, threshold=0.5):
        """
        Method evaluates model performance on provided data using:
            - f1 score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
            - logloss; https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
        :param X: dataframe to create predictions for
        :param y: true labels associated with given data
        :param threshold: for f1 score the probability threshold to consider class as positive.
            Should be in [0, 1] interval
        :return: dictionary of the form:
            {
                'f1_score': f1 score for given data,
                'logloss': logistic loss for given data
            }
        """

        # check input type
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f'Expected pandas DataFrame as input but {type(X)} is given')

        # check target consistency
        if len(set(y)) != 2 or len(set(y).intersection({0, 1})) != 2:
            raise ValueError('Invalid target provided: expected binary variable in {0, 1}')

        # check threshold consistency
        if not 0 <= threshold <= 1:
            raise ValueError(f'Invalid threshold {threshold}; should be in [0, 1] interval')

        y_fitted_p = self.predict_proba(X)[:, 1]  # get predictions from the model
        full_rows_mask = ~pd.isna(y_fitted_p)  # define rows with missing values
        y_fitted_p[~full_rows_mask] = -1  # replace missing values with technical -1 to perform threshold comparison
        y_fitted = np.where(y_fitted_p > threshold, 1, 0)  # apply threshold to obtain predicted classes

        # warning about estimate on subset of data
        if len(y_fitted[full_rows_mask]) < len(y):
            warnings.warn(f'Due to taken policies estimating performance on {len(y_fitted[full_rows_mask])} '
                          f'out of {len(y)} samples')

        return {
            'f1_score': metrics.f1_score(y[full_rows_mask], y_fitted[full_rows_mask]),
            'logloss': metrics.log_loss(y[full_rows_mask], y_fitted_p[full_rows_mask]),
        }

    def set_params(self, **params):
        """
        Set object parameters
        """

        valid_params = self.get_params().keys()
        model_params = self.model.get_params().keys()

        for key, value in params.items():

            if key not in valid_params:
                raise ValueError(f'Invalid parameter "{key}"')
            elif key in model_params:
                setattr(self, key, value)
                setattr(self.model, key, value)
            else:
                setattr(self, key, value)

    def tune_parameters(self, X, y, parameters, scoring=metrics.log_loss,
                        greater_is_better=False, needs_proba=True, cv=3, full_results=False):
        """
        Method performs parameters tuning following grid search cross-validation strategy
        :param X: features dataframe to use for parameters tuning
        :param y: target labels associated with X
        :param parameters: dictionary that defines a parameters grid to performa search over
            Example:
            {
                param1: (v1, v2, ...),
                param2: (v1, v2, ...),
                ...
            }
            For more details check https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        :param scoring: function object used for evaluating model performance. Can be sklearn metric or other custom
            function that takes true labels and predictions and returns difference measure
        :param greater_is_better: whether higher values of provided 'scoring' function imply better model performance
        :param needs_proba: whether provided 'scoring' function needs probabilities (otherwise classes) for calculations
        :param cv: number of cross-validation folds to use in parameters tuning
        :param full_results: whether to return full GridSearchCV statistics. If False then only set of best parameters
            along with best score will be returned
        :return: pandas dataframe with full search statistics if full_results=True, else dictionary with best
            patameters combination + best cv score
        """

        # wrap provided scoring function to allow missing values in predictions
        def score_wrap(y_true, y_pred):
            full_rows_mask = ~pd.isna(y_pred) if len(y_pred.shape) == 1 else ~np.any(pd.isna(y_pred), axis=1)
            return scoring(y_true[full_rows_mask], y_pred[full_rows_mask])

        # apply grid search procedure
        clf = GridSearchCV(self, parameters,
                           cv=cv,
                           scoring=make_scorer(score_wrap,
                                               greater_is_better=greater_is_better,
                                               needs_proba=needs_proba),
                           refit=False,
                           return_train_score=False
                           )

        clf.fit(X, y)

        # return results
        res = pd.DataFrame(clf.cv_results_)

        if full_results:
            return res
        else:

            best_params = res['params'][res['rank_test_score'] == 1].values[0]
            best_score = res['mean_test_score'][res['rank_test_score'] == 1].values[0]

            return {**best_params, 'best_score': best_score}


if __name__ == '__main__':

    # read data from file
    data = pd.read_csv('./Data/DR_Demo_Lending_Club_reduced.csv', sep=',', header=0)

    # drop poor features
    data.drop(['Id', 'collections_12_mths_ex_med', 'pymnt_plan', 'initial_list_status'], axis=1, inplace=True)

    # move target into separate variable
    y = data['is_bad'].values
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

    # store model coefficients for tests
    import pickle
    with open('./Data/ExpectedModelsCoefficients.p', 'wb') as out:
        pickle.dump(model.coef_, out)

