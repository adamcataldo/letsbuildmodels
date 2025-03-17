from lbm.datasets.time_series import TimeSeriesPreprocessor
import yfinance as yf

class ClosePrices():
    def __init__(self, ticker, start='2010-09-09', lookback=256, lookahead=1, 
                 tgt_lookback=0):
        df = yf.download(ticker, start=start)
        self.df = df['Close']
        self.lookback = lookback
        self.lookahead = lookahead
        self.tgt_lookback = tgt_lookback

    def get_loaders(self, batch_size=64):
        preprocessor = TimeSeriesPreprocessor(self.df, self.lookback,
                                              self.lookahead, self.tgt_lookback)
        return preprocessor.get_loaders(batch_size)