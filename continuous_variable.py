import pandas as pd
import numpy as np
import warnings


class ContinuousVariable:
    """
    This class implements continuous variable with the following functionality

    1. Replace special NA symbols e.g. 'unknown', 'na' etc and convert variable to float (will raise an exception
    if it is not possible for given input)

    2. Apply basic tests whether variable contains only NA or 0 variance i.e. comprised with one static value

    3. Deal with NA in data both for train and predict phase

    Parameters
    ----------
    variable_name : str, default: 'x'
        The name of the variable in text format

    na_policy : str, {'ignore', 'mean_replace', 'median_replace', 'na_mark_feature'}, default: 'median_replace'
        This parameter defines what to do with missing values in data:
        - 'ignore': leave missing values in variable as np.nan
        - 'mean_replace': calculate mean value of the variable based on known values and replace missing with mean
        - 'median_replace': calculate median value of the variable based on known values and replace missing with median
        - 'na_mark_feature': all missing values will be replaced with 0 and additional feature
            'variable_name_Unknown' will be populated. Additional feature will contain 1s where original variable
            had NA and 0 elsewhere.

    special_na : list (or other iterable collection) or None, default: None
        Parameter specifies a set of special symbols that should be treated as NA for given variable

    Attributes
    ----------
    fill : float or None
        This attribute stores value that should be used to replace missing values during prediction phase
        For 'ignore' na_policy it will contain None (no replacements), for 'na_mark_feature' - 0 and for
        'mean_replace'/'median_replace' mean and median of the feature correspondingly

    Example
    ----------

    # init variable
    var = ContinuousVariable(variable_name='x', na_policy='mean_replace', special_na={'na', 'unknown'})

    # specify train data to fit variable and test data to transform
    train = np.array([1, 2, 'na', 3, 'unknown'])
    test = np.array([4, 5, 'na'])

    var.fit(train)  # fit variable
    test_processed = var.transform(test)  # get test data processed
    print(test_processed)  # print results

    ---------
    {'x': array([4., 5., 2.])}

        In example above special symbol 'na' in test data was first replaced by ordinary np.nan and then replaced
    with mean value of present train observations i.e. mean(1, 2, 3)=2
    """

    @staticmethod
    def cast_to_float(x, variable_name):
        """
        Method converts given variable to float or rises an exception if conversion failed
        :param x: input variable to convert
        :param variable_name: name of the variable (to mention in error message if appear during conversion)
        :return: input variable with np.float64 type
        """

        try:
            x = np.array(x).astype(np.float64)
            return x
        except ValueError:
            raise ValueError(f'Variable "{variable_name}" is specified as numeric but conversion to float is failed. '
                             'Try to specify it as categorical or check whether it includes special N/A symbols')

    @staticmethod
    def replace_special_na(x, special_na=None):
        """
        Method replaces all special symbols with np.nan
        :param x: input variable to make replacements in
        :param special_na: a list of special symbols to replace (if None -> make no replacements)
        :return: input variable with all special symbols replaced with np.nan
        """

        if special_na is not None:
            return np.array(list(map(lambda e: np.nan if e in special_na else e, x)))
        else:
            return x

    @staticmethod
    def check_variance(x, variable_name):
        """
        Method checks whether given variable comprised only with missing values or has 0 variance
        :param x: input variable to test
        :param variable_name: name of the variable to mention in warnings
        """

        if np.all(pd.isna(x)):
            warnings.warn(f'Variable {variable_name} contains only empty values')
        elif np.var(x[~pd.isna(x)]) == 0:
            warnings.warn(f'Variable {variable_name} has constant value')

    def __init__(self, variable_name='x', na_policy='median_replace', special_na=None):

        self.na_policy = na_policy
        self.special_na = special_na
        self.fill = None
        self.is_fitted = False
        self.variable_name = variable_name

    def fit(self, x, y=None):
        """
        Fit the variable according to provided data
        The following steps will be performed:
            - replace special NA based on provided 'special_na' parameter
            - convert values to float type (or raise an error)
            - check variance of the variable
            - calculate filling value for missing values based on provided 'na_policy' parameter
        :param x: input variable values
        :param y: None (target is not needed and added for compatibility)
        :return: self
        """

        # replace special na
        x = ContinuousVariable.replace_special_na(x, self.special_na)
        # convert to float
        x = ContinuousVariable.cast_to_float(x, self.variable_name)
        # check variance
        ContinuousVariable.check_variance(x, self.variable_name)

        # calculate and store missing fill value based on provided policy
        if self.na_policy == 'ignore':
            self.fill = None
        elif self.na_policy == 'mean_replace':
            self.fill = np.mean(x[~pd.isna(x)])
        elif self.na_policy == 'median_replace':
            self.fill = np.median(x[~pd.isna(x)])
        elif self.na_policy == 'na_mark_feature':
            self.fill = 0.
        else:
            raise ValueError(f'Not valid n/a policy "{self.na_policy}" for continuous variable "{self.variable_name}"')

        self.is_fitted = True

        return self

    def transform(self, x):
        """
        Prepare provided values for inference based on feature parameters
        :param x: values to process
        :return: dictionary of the form: {variable_name -> values}
        """

        # default transformation under 'ignore' na policy can be performed without previous 'fit'
        # in case of unexpected behaviour through a warning about underlying policy
        if not self.is_fitted:
            warnings.warn(f'Running transform before fit for variable "{self.variable_name}"; '
                          'Default "ignore" policy will be applied')

        # replace special na
        x = ContinuousVariable.replace_special_na(x, self.special_na)
        # convert to float
        x = ContinuousVariable.cast_to_float(x, self.variable_name)

        # fill na and format variable according to specified policy and output format
        if self.fill is None:
            # do nothing if 'ignore'
            return {self.variable_name: x}
        elif self.na_policy in {'mean_replace', 'median_replace'}:
            # replace missing values with 'self.fill'
            return {self.variable_name: np.where(pd.isna(x), self.fill, x)}
        elif self.na_policy == 'na_mark_feature':
            # replace missing values with 'self.fill' and create additional variable to mark NA
            return {
                self.variable_name: np.where(pd.isna(x), self.fill, x),
                f'{self.variable_name}_Unknown': np.where(pd.isna(x), 1, 0)
            }
        else:
            raise ValueError(f'Not valid n/a policy "{self.na_policy}" for continuous variable "{self.variable_name}"')

    def fit_transform(self, x, y=None):
        """
        Apply fit method and return transformed feature
        """

        self.fit(x, y)
        return self.transform(x)
