import unittest
import pandas as pd
from lbm.tranformations.returns import get_returns, apply_returns
from pandas import DatetimeIndex

class TestReturns(unittest.TestCase):
    def test_get_returns(self):
        df = pd.DataFrame({'price': [100.0, 110.0, 121.0]})
        result = get_returns(df)
        expected = pd.DataFrame({'price': [None, 0.1, 0.1]})
        pd.testing.assert_frame_equal(result, expected)

    def test_get_returns_preserves_index(self):
        df = pd.DataFrame({'price': [100.0, 110.0, 121.0]})
        idx = DatetimeIndex(['2025-01-01', '2025-01-02', '2025-01-03'])
        df.index = idx
        result = get_returns(df)
        expected = pd.DataFrame({'price': [None, 0.1, 0.1]})
        expected.index = idx
        pd.testing.assert_frame_equal(result, expected)

    def test_apply_returns(self):
        returns_df = pd.DataFrame({'returns': [None, -0.5, 0.2]})
        result = apply_returns(200.0, returns_df)
        expected = pd.DataFrame({'returns': [200.0, 100.0, 120.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_apply_returns_preserves_index(self):
        returns_df = pd.DataFrame({'returns': [None, -0.5, 0.2]})
        idx = DatetimeIndex(['2025-01-01', '2025-01-02', '2025-01-03'])
        returns_df.index = idx
        result = apply_returns(200.0, returns_df)
        expected = pd.DataFrame({'returns': [200.0, 100.0, 120.0]})
        expected.index = idx
        pd.testing.assert_frame_equal(result, expected)    

if __name__ == '__main__':
    unittest.main()