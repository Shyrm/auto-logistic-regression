import pandas as pd
import numpy as np
import warnings
import category_encoders as ce
from sklearn.exceptions import NotFittedError


class CategoricalVariable:
    """
    This class implements categorical variable with the following functionality

    1. Replace special NA symbols e.g. 'unknown', 'na' etc.

    2. Apply basic tests:
        - whether variable contains only NA
        - whether variable contains only one category
        - whether variable's cardinality is too high (number of unique categories greater than 0.5 * dataset_size)
        The latter is undesired for one-hot encoding policy

    3. Deal with NA in data both for train and predict phase

    4. Encode categorical variable following specified policy

    Parameters
    ----------
    variable_name : str, default: 'x'
        The name of the variable in text format

    na_policy : str, {'ignore', 'new_class'}, default: 'new_class'
        This parameter defines what to do with missing values in data:
        - 'ignore': leave missing values in variable as np.nan
        - 'new_class': treat all missing values as separate category (class) of the variable

    special_na : list (or other iterable collection) or None, default: None
        Parameter specifies a set of special symbols that should be treated as NA for given variable

    encoding_policy : str, {'ohe', 'target_encoding'}, default: 'ohe'
        Parameter specifies the way to encode categorical variable
        - 'ohe': one-hot encoding (http://contrib.scikit-learn.org/categorical-encoding/onehot.html)
        - 'target_encoding': target encoding (http://contrib.scikit-learn.org/categorical-encoding/targetencoder.html)

    new_category_policy : 'str', {'ignore', 'pass'}, default: 'pass'
        Parameter specifies what to do with new category in data
        - 'ignore': replace new category with np.nan
        - 'pass': allow new category.
        If 'pass' policy is chosen then depending on 'encoding_policy' new category will appear as follows
        1) for 'ohe' all dummy features of known categories will take 0s
        2) for 'target_encoding' global target average will be assigned for new category

    smoothing : float, default: 1.0
        Smoothing effect to balance categorical average vs prior.
        Applicable only for 'target_encoding' policy
        For more details check http://contrib.scikit-learn.org/categorical-encoding/targetencoder.html

    min_samples_leaf : int, default: 1
        Minimum samples to take category average into account
        Applicable only for 'target_encoding' policy
        For more details check http://contrib.scikit-learn.org/categorical-encoding/targetencoder.html

    Attributes
    ----------
    encoder : encoder instance from 'category_encoders' module

        For 'ohe' policy will be category_encoders.one_hot.OneHotEncoder
        For 'target_encoding' policy will be category_encoders.target_encoder.TargetEncoder
        For more details check http://contrib.scikit-learn.org/categorical-encoding/

    Example
    ----------
    # init variable with new category and na policies
    var = CategoricalVariable(na_policy='new_class',
                              new_category_policy='pass',
                              encoding_policy='ohe',
                              special_na={'Unknown'})

    # train input data
    train_input = pd.DataFrame(
        data={
            'x': ['a', 'a', 'b', 'b', 'Unknown', 'Unknown'],
            'y': [1, 0, 1, 0, 1, 0]
        }
    )

    # fit variable with train data
    var.fit(train_input['x'], train_input['y'])

    # get transformed validation
    valid = var.transform(['a', 'b', 'c'])

    # print results
    print(valid)

    ----------
    {'x_ca': array([1, 0, 0]), 'x_cb': array([0, 1, 0]), 'x_nan': array([0, 0, 0])}

        In example above 3 dummy variables were populated due to 'ohe' encoding policies: for class 'a' -> 'x_ca'
    for class 'b' -> 'x_cb' and for 'Unknown' -> 'x_nan'. New category 'c' that were not observed in train data
    is not present in 'valid' but indirectly marked by setting 0s to all the rest dummy features.

    """

    @staticmethod
    def replace_special_na(x, special_na=None):
        """
        Method replaces all special symbols with np.nan
        :param x: input variable to make replacements in
        :param special_na: a list of special symbols to replace (if None -> make no replacements)
        :return: input variable with all special symbols replaced with np.nan
        """
        if special_na is not None:
            return list(map(lambda e: np.nan if e in special_na else e, x))
        else:
            return x.values

    @staticmethod
    def check_variance(x, variable_name):
        """
        Method applies 3 checks on input variable:
            - whether variable contains only NA
            - whether variable contains only one category
            - whether variable's cardinality is too high (number of unique categories greater than 0.5 * dataset_size)
        :param x: input variable to test
        :param variable_name: name of the variable to mention in warnings
        """

        if np.all(pd.isna(x)):
            warnings.warn(f'Variable {variable_name} contains only empty values')
        else:
            card = len(set(x))
            if card == 1:
                warnings.warn(f'Variable {variable_name} has only one class')
            elif card / len(x) > 0.5:
                warnings.warn(f'Variable "{variable_name}" has too high cardinality '
                              'for ordinary one-hot encoding policy')

    @staticmethod
    def convert_to_string(x, variable_name):
        """
        Method converts input variable into string format
        Variable will appear as one column pandas dataframe.
        All missing values will be replaced with np.nan
        All variable categories will get 'c' prefix to ensure correct conversion
        :param x: input variable to convert
        :param variable_name: name of the variable (will be used as column name in resulting dataframe)
        :return: pandas dataframe with variable values
        """

        return pd.DataFrame({variable_name: list(map(lambda e: np.nan if pd.isna(e) else f'c{e}', x))})

    def __init__(self, variable_name='x', na_policy='new_class', special_na=None,
                 encoding_policy='ohe', new_category_policy='pass',
                 min_samples_leaf=1, smoothing=1.):

        self.na_policy = na_policy
        self.special_na = special_na
        self.variable_name = variable_name
        self.encoding_policy = encoding_policy
        self.new_category_policy = new_category_policy
        self.encoder = None
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing

    def _init_encoder(self):
        """
        Method checks whether provided policies are valid and define which encoder to init for current variable
        Once the encoder and policies are defined -> assign created encoder to self.encoder
        """

        # set na policy
        if self.na_policy == 'ignore':
            handle_missing = 'return_nan'
        elif self.na_policy == 'new_class':
            handle_missing = 'value'
        else:
            raise ValueError(f'Invalid na policy {self.na_policy} for categorical variable {self.variable_name}')

        # set new category policy
        if self.new_category_policy == 'ignore':
            handle_unknown = 'return_nan'
        elif self.new_category_policy == 'pass':
            handle_unknown = 'value'
        else:
            raise ValueError(f'Invalid new category policy {self.new_category_policy} '
                             f'for categorical variable {self.variable_name}')

        # init encoder based on variable parameters
        if self.encoding_policy == 'ohe':

            self.encoder = ce.one_hot.OneHotEncoder(
                handle_missing=handle_missing,
                handle_unknown=handle_unknown,
                use_cat_names=True
            )

        elif self.encoding_policy == 'target_encoding':

            self.encoder = ce.target_encoder.TargetEncoder(
                handle_missing=handle_missing,
                handle_unknown=handle_unknown,
                min_samples_leaf=self.min_samples_leaf,
                smoothing=self.smoothing
            )

        else:
            raise ValueError(f'Invalid encoding policy {self.encoding_policy} for '
                             f'categorical variable {self.variable_name}')

    def fit(self, x, y=None):
        """
        Fit the variable according to provided data
        The following steps will be performed:
            - replace special NA based on provided 'special_na' parameter
            - check whether variable has only missing values
            - check whether variable has only one class
            - check whether variable has too high cardinality (undesired for 'ohe' policy)
            - initiate and fit underlying encoder based on specified policies
        :param x: input variable values
        :param y: target variable values (needed for 'target_encoding' policy)
        :return:
        """

        # replace special na
        x = CategoricalVariable.replace_special_na(x, self.special_na)
        # apply variance tests
        CategoricalVariable.check_variance(x, self.variable_name)
        # convert variable to string
        x = CategoricalVariable.convert_to_string(x, self.variable_name)

        # init and fit encoder
        self._init_encoder()
        if self.encoding_policy == 'ohe':
            self.encoder.fit(x)
        elif y is None:
            raise ValueError(f'Target should be provided for categorical encoding policy '
                             f'(variable name "{self.variable_name}")')
        else:
            self.encoder.fit(x, y)

        return self

    def transform(self, x):
        """
        Prepare provided values for inference based on feature parameters
        :param x: values to process
        :return: dictionary of the form: {variable_name -> values}
        """

        # check whether variable was fitted before
        if self.encoder is None:
            raise NotFittedError(f'Attempt to transform before "fit" for variable {self.variable_name}')

        # replace special na and convert to string
        x = CategoricalVariable.replace_special_na(x, self.special_na)
        x = CategoricalVariable.convert_to_string(x, self.variable_name)

        # run transform in underlying encoder
        if self.encoding_policy == 'ohe':
            trf = self.encoder.transform(x)

            # for one-hot encoding policy if 'ignore' is taken then dummy feature 'variable_name_nan' is redundant
            # since all missing values will be marked as np.nan. We drop it before returning results
            if self.na_policy == 'ignore':
                trf.drop(f'{self.variable_name}_nan', axis=1, inplace=True)

            return {col: trf[col].values for col in trf.columns}
        else:
            return {self.variable_name: self.encoder.transform(x).iloc[:, 0].values}

    def fit_transform(self, x, y=None):
        """
        Apply fit method and return transformed feature
        """

        self.fit(x, y)
        return self.transform(x)




