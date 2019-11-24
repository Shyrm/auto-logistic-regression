import unittest
import numpy as np
import pandas as pd
from continuous_variable import ContinuousVariable
from categorical_variable import CategoricalVariable


class TestVariable(unittest.TestCase):

    @staticmethod
    def nan_equal(a, b):
        try:
            np.testing.assert_equal(a, b)
        except AssertionError:
            return False
        return True

    def compare_variables(self, variables, expected_outputs, x_train, x_valid, y_train=None):

        if len(variables) != len(expected_outputs):
            raise ValueError('Number of provided variables must be equal to number of expected outputs '
                             f'but {len(variables)} variables and {len(expected_outputs)} outputs were given')

        for var, out in zip(variables, expected_outputs):

            # fit variable with designed input
            var.fit(x_train, y_train)

            # get transformed validation input
            actual_out = var.transform(x_valid)

            self.assertEqual(len(actual_out), len(out))

            # compare expected and actual outputs
            for k, expected_out in out.items():
                self.assertIn(k, actual_out.keys())
                self.assertTrue(TestVariable.nan_equal(expected_out, actual_out[k]))

    def testCategoricalNAReplace(self):

        # init different categorical variables with different na and encoding policies
        variables = [
            CategoricalVariable(na_policy='ignore', encoding_policy='ohe', special_na={'Unknown'}),
            CategoricalVariable(na_policy='new_class', encoding_policy='ohe', special_na={'Unknown'}),
            CategoricalVariable(na_policy='ignore', encoding_policy='target_encoding', special_na={'Unknown'}),
            CategoricalVariable(na_policy='new_class', encoding_policy='target_encoding', special_na={'Unknown'}),
        ]

        # define train and validation inputs for testing
        train_input = pd.DataFrame(
            data={
                'x': ['a', 'a', 'b', 'b', 'Unknown', 'Unknown'],
                'y': [1, 0, 1, 0, 1, 0]
            }
        )

        validation_input = pd.DataFrame(data={'x': ['a', 'b', 'Unknown']})

        # define expected outputs
        outputs = [
            {'x_ca': np.array([1, 0, np.nan]), 'x_cb': np.array([0, 1, np.nan])},
            {'x_ca': np.array([1, 0, 0]), 'x_cb': np.array([0, 1, 0]), 'x_nan': np.array([0, 0, 1])},
            {'x': np.array([0.5, 0.5, np.nan])},
            {'x': np.array([0.5, 0.5, 0.5])}
        ]

        # run tests
        self.compare_variables(variables, outputs,
                               x_train=train_input['x'],
                               x_valid=validation_input['x'],
                               y_train=train_input['y'])

    def testCategoricalNewClass(self):

        # init different categorical variables with different new category and encoding policies
        variables = [
            CategoricalVariable(new_category_policy='ignore', encoding_policy='ohe', special_na={'Unknown'}),
            CategoricalVariable(new_category_policy='pass', encoding_policy='ohe', special_na={'Unknown'}),
            CategoricalVariable(new_category_policy='ignore', encoding_policy='target_encoding', special_na={'Unknown'}),
            CategoricalVariable(new_category_policy='pass', encoding_policy='target_encoding', special_na={'Unknown'}),
        ]

        # define train and validation inputs for testing
        train_input = pd.DataFrame(
            data={
                'x': ['a', 'a', 'b', 'b', 'Unknown', 'Unknown'],
                'y': [1, 0, 1, 0, 1, 0]
            }
        )

        validation_input = pd.DataFrame(data={'x': ['a', 'b', 'c']})

        # define expected outputs
        outputs = [
            {'x_ca': np.array([1, 0, np.nan]), 'x_cb': np.array([0, 1, np.nan]), 'x_nan': np.array([0, 0, np.nan])},
            {'x_ca': np.array([1, 0, 0]), 'x_cb': np.array([0, 1, 0]), 'x_nan': np.array([0, 0, 0])},
            {'x': np.array([0.5, 0.5, np.nan])},
            {'x': np.array([0.5, 0.5, 0.5])}
        ]

        # run tests
        self.compare_variables(variables, outputs,
                               x_train=train_input['x'],
                               x_valid=validation_input['x'],
                               y_train=train_input['y'])

    def testContinuousNAReplace(self):

        # init different continuous variables with different na policies
        variables = [
            ContinuousVariable(na_policy='ignore', special_na={'na', 'unknown'}),
            ContinuousVariable(na_policy='mean_replace', special_na={'na', 'unknown'}),
            ContinuousVariable(na_policy='median_replace', special_na={'na', 'unknown'}),
            ContinuousVariable(na_policy='na_mark_feature', special_na={'na', 'unknown'}),
        ]

        # define train and validation inputs for testing
        train_input = np.array([1, 2, 'na', 3, 'unknown'])
        validation_input = np.array([4, 5, 'na'])

        # define expected outputs
        outputs = [
            {'x': np.array([4, 5, np.nan])},
            {'x': np.array([4, 5, 2])},
            {'x': np.array([4, 5, 2])},
            {'x': np.array([4, 5, 0]), 'x_Unknown': np.array([0, 0, 1])}
        ]

        # run tests
        self.compare_variables(variables, outputs, x_train=train_input, x_valid=validation_input, y_train=None)


if __name__ == '__main__':

    unittest.main()
