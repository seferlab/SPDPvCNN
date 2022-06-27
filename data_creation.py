import yfinance as yf
import pandas as pd
import numpy as np
import talib as tb

'''
DEFINING SOME VARIABLES
'''
startDate = '2001-10-11'
endDate = '2022-04-15'
axes = ['Date', 'Value']
headers = ['RSI', 'CMO', 'PLUS_DI', 'MINUS_DI', 'WILLR', 'CCI', 'ULTOSC', 'AROONOSC', 'MFI', 'MOM', 'MACD', 'MACDFIX', 'LINEARREG_ANGLE', 'LINEARREG_SLOPE', 'ROCP', 'ROC', 'ROCR', 'ROCR100', 'SLOWK',
           'FASTD', 'SLOWD', 'AROONUP', 'AROONDOWN', 'APO', 'MACDEXT', 'FASTK', 'PPO', 'MINUS_DM', 'ADOSC', 'FASTDRSI', 'FASTKRSI', 'TRANGE', 'TRIX', 'STD', 'BOP', 'VAR', 'PLUS_DM', 'CORREL', 'AD',
           'BETA', 'WCLPRICE', 'TSF', 'TYPPRICE', 'AVGPRICE', 'MEDPRICE', 'BBANDSL', 'LINEARREG', 'OBV', 'BBANDSM', 'TEMA', 'BBANDSU', 'DEMA', 'MIDPRICE', 'MIDPOINT', 'WMA', 'EMA',
           'HT_TRENDLINE', 'KAMA', 'SMA', 'MA', 'ADXR', 'ADX', 'TRIMA', 'LINEARREG_INTERCEPT', 'DX']


etfList = ['XLF', 'XLU', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']
threshold = 0.01  # Re-arrange the Threshold Value

pd.set_option('display.max_rows', None)

'''
DOWNLOADING THE DATA
'''
# DataFrame, size=(n_days, 6), col_names=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
for etf in etfList:

    imageList = []
    labelList = []

    data = yf.download(etf, start=startDate, end=endDate)

    '''
    CALCULATING THE INDICATOR VALUES
    '''
    # DataFrame, size=(n_days, 2), col_names=["Date", "Value"]
    rsi = tb.RSI(data["Close"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    wma = tb.WMA(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    ema = tb.EMA(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    sma = tb.SMA(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    roc = tb.ROC(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    cmo = tb.CMO(data["Close"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    cci = tb.CCI(data["High"], data["Low"], data["Close"],
                 timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    ppo = tb.PPO(data["Close"], fastperiod=12, slowperiod=26,
                 matype=0).to_frame().reset_index().set_axis(axes, axis=1)
    tema = tb.TEMA(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    willr = tb.WILLR(data["High"], data["Low"], data["Close"],
                     timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    macd, macdsignal, macdhist = tb.MACD(
        data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    macd = macd.to_frame().reset_index().set_axis(axes, axis=1)

    sar = tb.SAR(data["High"], data["Low"], acceleration=0,
                 maximum=0).to_frame().reset_index().set_axis(axes, axis=1)
    adx = tb.ADX(data["High"], data["Low"], data["Close"],
                 timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    std = tb.STDDEV(data['Close'], timeperiod=5, nbdev=1).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    obv = tb.OBV(data['Close'], data['Volume']).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    adxr = tb.ADXR(data["High"], data["Low"], data["Close"],
                   timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    apo = tb.APO(data['Close'], fastperiod=12, slowperiod=26,
                 matype=0).to_frame().reset_index().set_axis(axes, axis=1)
    aroondown, aroonup = tb.AROON(data["High"], data["Low"], timeperiod=14)
    aroondown = aroondown.to_frame().reset_index().set_axis(axes, axis=1)
    aroonup = aroonup.to_frame().reset_index().set_axis(axes, axis=1)
    aroonosc = tb.AROONOSC(data["High"], data["Low"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    bop = tb.BOP(data["Open"], data["High"], data["Low"], data["Close"]).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    dx = tb.DX(data["High"], data["Low"], data["Close"],
               timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    macdext, macdextsignal, macdexthist = tb.MACDEXT(
        data["Close"], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    macdext = macdext.to_frame().reset_index().set_axis(axes, axis=1)
    macdfix, macdfixsignal, macdfixhist = tb.MACDFIX(
        data["Close"], signalperiod=9)
    macdfix = macdfix.to_frame().reset_index().set_axis(axes, axis=1)
    mfi = tb.MFI(data["High"], data["Low"], data["Close"], data["Volume"],
                 timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    minus_di = tb.MINUS_DI(data["High"], data["Low"], data["Close"],
                           timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    minus_dm = tb.MINUS_DM(data["High"], data["Low"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    mom = tb.MOM(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    plus_di = tb.PLUS_DI(data["High"], data["Low"], data["Close"],
                         timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    plus_dm = tb.PLUS_DM(data["High"], data["Low"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    rocp = tb.ROCP(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    rocr = tb.ROCR(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    rocr100 = tb.ROCR100(data["Close"], timeperiod=10).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    slowk, slowd = tb.STOCH(data["High"], data["Low"], data["Close"], fastk_period=5,
                            slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    slowk = slowk.to_frame().reset_index().set_axis(axes, axis=1)
    slowd = slowd.to_frame().reset_index().set_axis(axes, axis=1)
    fastk, fastd = tb.STOCHF(
        data["High"], data["Low"], data["Close"], fastk_period=5, fastd_period=3, fastd_matype=0)
    fastk = fastk.to_frame().reset_index().set_axis(axes, axis=1)
    fastd = fastd.to_frame().reset_index().set_axis(axes, axis=1)
    fastkrsi, fastdrsi = tb.STOCHRSI(
        data["Close"], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    fastkrsi = fastkrsi.to_frame().reset_index().set_axis(axes, axis=1)
    fastdrsi = fastdrsi.to_frame().reset_index().set_axis(axes, axis=1)
    trix = tb.TRIX(data["Close"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    ultosc = tb.ULTOSC(data["High"], data["Low"], data["Close"], timeperiod1=7,
                       timeperiod2=14, timeperiod3=28).to_frame().reset_index().set_axis(axes, axis=1)

    bbands_upperband, bbands_middleband, bbands_lowerband = tb.BBANDS(
        data['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    bbands_upperband = bbands_upperband.to_frame().reset_index().set_axis(axes, axis=1)
    bbands_middleband = bbands_middleband.to_frame().reset_index().set_axis(axes, axis=1)
    bbands_lowerband = bbands_lowerband.to_frame().reset_index().set_axis(axes, axis=1)
    dema = tb.DEMA(data['Close'], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    ht_trendline = tb.HT_TRENDLINE(
        data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    kama = tb.KAMA(data['Close'], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    ma = tb.MA(data['Close'], timeperiod=30, matype=0).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    midpoint = tb.MIDPOINT(data['Close'], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    midprice = tb.MIDPRICE(data["High"], data["Low"], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    sarext = tb.SAREXT(data["High"], data["Low"], startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0,
                       accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0).to_frame().reset_index().set_axis(axes, axis=1)
    trima = tb.TRIMA(data['Close'], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    ad = tb.AD(data["High"], data["Low"], data['Close'],
               data['Volume']).to_frame().reset_index().set_axis(axes, axis=1)
    adosc = tb.ADOSC(data["High"], data["Low"], data['Close'], data['Volume'],
                     fastperiod=3, slowperiod=10).to_frame().reset_index().set_axis(axes, axis=1)

    trange = tb.TRANGE(data["High"], data["Low"], data['Close']).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    avgprice = tb.AVGPRICE(data['Open'], data["High"], data["Low"],
                           data['Close']).to_frame().reset_index().set_axis(axes, axis=1)
    medprice = tb.MEDPRICE(data["High"], data["Low"]).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    typprice = tb.TYPPRICE(data["High"], data["Low"], data['Close']).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    wclprice = tb.WCLPRICE(data["High"], data["Low"], data['Close']).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    beta = tb.BETA(data["High"], data["Low"], timeperiod=5).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    correl = tb.CORREL(data["High"], data["Low"], timeperiod=30).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    linearreg = tb.LINEARREG(data['Close'], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    linearreg_angle = tb.LINEARREG_ANGLE(
        data['Close'], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    linearreg_intercept = tb.LINEARREG_INTERCEPT(
        data['Close'], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    linearreg_slope = tb.LINEARREG_SLOPE(
        data['Close'], timeperiod=14).to_frame().reset_index().set_axis(axes, axis=1)
    tsf = tb.TSF(data['Close'], timeperiod=14).to_frame(
    ).reset_index().set_axis(axes, axis=1)
    var = tb.VAR(data['Close'], timeperiod=5, nbdev=1).to_frame(
    ).reset_index().set_axis(axes, axis=1)

    '''
    PREPROCESSING INDICATOR DATA
    '''
    # List of (indicators) DataFrames, size=n_indicators
    indicators = [rsi, cmo, plus_di, minus_di, willr, cci, ultosc, aroonosc, mfi, mom, macd, macdfix, linearreg_angle, linearreg_slope, rocp, roc, rocr, rocr100, slowk, fastd, slowd, aroonup, aroondown, apo,
                  macdext, fastk, ppo, minus_dm, adosc, fastdrsi, fastkrsi, trange, trix, std, bop, var, plus_dm, correl, ad, beta, wclprice, tsf, typprice, avgprice, medprice, bbands_lowerband, linearreg, obv,
                  bbands_middleband, tema, bbands_upperband, dema, midprice, midpoint, wma, ema, ht_trendline, kama, sma, ma, adxr, adx, trima, linearreg_intercept, dx]
    # 15x15 matrix of indicators
    # [rsi, cmo, willr, cci, macd, roc, ppo, std, tema, obv, wma, ema, sma, adx, sar]

    # Number of indicators (int)
    nIndicators = len(indicators)

    # Calculating the most number of null values in an indicator DataFrame's "Value" column
    maxNullVal = -1
    for indicator in indicators:
        if(indicator['Value'].isnull().sum() > maxNullVal):
            maxNullVal = indicator['Value'].isnull().sum()

    # List of (indicators "Value" column) DataFrames, size=n_indicators
    indicatorValues = []
    for indicator in indicators:
        # Getting rid of null values
        indicatorValues.append(indicator['Value'].iloc[maxNullVal:])

    # DataFrame, size=(n_days, n_indicators, col_names=headers)
    indicatorValuesMatrix = pd.concat(indicatorValues, axis=1, keys=headers)
    indicatorCorr = indicatorValuesMatrix.corr(method='pearson')

    '''
    dictCor = {}
    for header, value in zip(headers, indicatorCorr.iloc[0]):
        dictCor[header] = value
    sortedDictCor = {k: v for k, v in sorted(dictCor.items(), key=lambda item: abs(item[1]), reverse=True)}
    for k,v in sortedDictCor.items():
        print(k, v)

    '''

    '''
    CREATING THE IMAGES
    '''
    # nDays = len(indicatorValues[0])
    # for idx in range(nDays-nIndicators):
    #     # List, size=n_indicators, contains imageRows of size (n_indicators, 1)
    #     image = []
    #     for indicatorValue in indicatorValues:
    #         # NumPy Array, size=(n_indicators, 1)
    #         imageRow = indicatorValue[idx:idx+nIndicators][..., np.newaxis]
    #         image.append(imageRow)
    #     imageList.append(np.array(image))

    nDays = len(indicatorValues[0])
    for idx in range(nDays-2*nIndicators):
        # List, size=n_indicators, contains imageRows of size (n_indicators, 1)
        image = []
        for indicatorValue in indicatorValues:
            # NumPy Array, size=(n_indicators, 1)
            imageRow = indicatorValue[idx:idx+2*nIndicators].values
            image.append(imageRow)
        imageList.append(np.array(image))

    '''
    CREATING THE LABELS
    '''
    # Pandas Series, size=n_days-(maxNullVal+nIndicators-1) -> Check this, size is imageList+1, might be a bug.
    data_close = data[maxNullVal+2*nIndicators-1:]["Close"]

    # Buy : 0
    # Hold: 1
    # Sell: 2
    for i in range(len(data_close)-1):
        closePriceDifference = data_close.iloc[i+1] - data_close.iloc[i]
        thresholdPrice = threshold * data_close.iloc[i]
        # If the price has increased
        if(closePriceDifference > 0):
            # but not enough to pass the threshold
            if(closePriceDifference <= thresholdPrice):
                labelList.append(np.array([1.0]))  # HOLD
            # enough to pass the threshold
            else:
                labelList.append(np.array([0.0]))  # BUY
        # If the price has decreased
        elif(closePriceDifference < 0):
            # but not so much to pass the thresshold
            if(abs(closePriceDifference) <= thresholdPrice):
                labelList.append(np.array([1.0]))  # HOLD
            # so much to pass the threshold
            else:
                labelList.append(np.array([2.0]))  # SELL
        # If the price hasn't changed
        else:
            labelList.append(np.array([1.0]))  # HOLD

    print(len(imageList))
    print(len(labelList))
    print(len(data_close[:-1]))

    # imageList = np.array(imageList)
    # labelList = np.array(labelList)

    # unique, counts = np.unique(labelList, return_counts=True)
    # print(np.asarray((unique, counts)).T)

    # imageList_copy = imageList[:]
    # imageList_copy = imageList_copy.reshape(len(imageList), -1)
    # # df_before = pd.DataFrame(imageList_copy, columns=np.repeat(
    # #     np.array(headers), nIndicators))
    # # df_before.to_csv("df_before.csv", encoding='utf-8', index=False)
    # mean = np.mean(imageList_copy, axis=0)
    # # mean_df = pd.DataFrame(mean)
    # # mean_df.to_csv("mean.csv", encoding='utf-8', index=False)
    # std = np.std(imageList_copy, axis=0)
    # # std_df = pd.DataFrame(std)
    # # std_df.to_csv("std.csv", encoding='utf-8', index=False)
    # imageList_copy = (imageList_copy - mean) / std
    # # df_after = pd.DataFrame(imageList_copy, columns=np.repeat(
    # #     np.array(headers), nIndicators))
    # # df_after.to_csv("df_after.csv", encoding='utf-8', index=False)
    # imageList = imageList_copy.reshape(
    #     len(imageList), len(indicators), len(indicators), 1)
    standartized_image_list = []
    for img in imageList:
        m = np.mean(img, axis=1, keepdims=True)
        s = np.std(img, axis=1, keepdims=True)
        standartized_image = np.expand_dims((img - m) / s, axis=-1)
        standartized_image_list.append(standartized_image)

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    train_date = []
    test_date = []

    train_price = []
    test_price = []

    for index in range(len(standartized_image_list)):
        if(index < (len(standartized_image_list) * 0.8)):
            x_train.append(standartized_image_list[index])
            y_train.append(labelList[index])
            train_date.append(data_close.index[index])
            train_price.append(data_close.iloc[index])
        else:
            x_test.append(standartized_image_list[index])
            y_test.append(labelList[index])
            test_date.append(data_close.index[index])
            test_price.append(data_close.iloc[index])

    np.save(f"./ETF/rectangle/01/TrainData/x_{etf}.npy", x_train)
    np.save(f"./ETF/rectangle/01/TrainData/y_{etf}.npy", y_train)
    np.save(f"./ETF/rectangle/01/TestData/x_{etf}.npy", x_test)
    np.save(f"./ETF/rectangle/01/TestData/y_{etf}.npy", y_test)

    np.save(f"./ETF/rectangle/01/Date/TrainDate/{etf}.npy", train_date)
    np.save(f"./ETF/rectangle/01/Date/TestDate/{etf}.npy", test_date)
    np.save(f'./ETF/rectangle/01/Price/TrainPrice/{etf}.npy', train_price)
    np.save(f'./ETF/rectangle/01/Price/TestPrice/{etf}.npy', test_price)
