import numpy as np
import pandas as pd

def main():
    do_exercise()

def do_exercise():
    # 1
    odf = pd.read_csv('AAPL.csv')
    # 2
    date_index = odf.pop('Date')
    odf.index = pd.to_datetime(date_index)
    # 3
    ndf = odf[['Open', 'Close', 'Volume']]
    # 4
    df = ndf[:]['1989' : '2003-04']
    return df

if __name__ == "__main__":
    main()