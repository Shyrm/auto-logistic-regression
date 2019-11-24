import unittest
import pickle
import pandas as pd
import numpy as np
from logistic_regression import LogisticRegressionWrapper

MODEL_COEFFICIENTS_FILE = './Data/ExpectedModelsCoefficients.p'
DATA_FILE = './Data/DR_Demo_Lending_Club_reduced.csv'


class TestModel(unittest.TestCase):

    def testReproducibility(self):

        # read expected coefficients from file
        with open(MODEL_COEFFICIENTS_FILE, 'rb') as src:
            target_coef = pickle.load(src)

        # fit model with provided data and obtain actual coefficients

        # read data from file
        data = pd.read_csv(DATA_FILE, sep=',', header=0)

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

        # fit model using all data and get actual coefficients
        model.fit(data, y)
        actual_coef = model.coef_

        # test whether target and actual coefficients match withing small tolerance
        # (small difference can appear depending on platform)
        self.assertTrue(np.allclose(target_coef, actual_coef))


if __name__ == '__main__':

    unittest.main()
