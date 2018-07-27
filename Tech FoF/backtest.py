import pandas as pd
import numpy as np
import sklearn
import sklearn.cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from itertools import combinations
import datetime
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from pprint import pprint
from sklearn.metrics import f1_score, r2_score
from scipy.stats import norm
from collections import defaultdict
from sqlalchemy import create_engine
import os
from sqlalchemy.orm import sessionmaker, scoped_session

if __name__ == "__main__":

	engine = create_engine('mysql://root:0000@localhost/bloomberg_prices')
	Session = scoped_session(sessionmaker(bind=engine))
	tables = list(Session().execute('SHOW TABLES'))
	tables = list(zip(*tables))[0]

	#Asking user for category input
	while True:
		user_input = input('Enter category for funds (core, satellite, etf, indiafof): ')
		if user_input not in tables:
		  print('Category unavailable. Please check SQL database for available prices.')
		  continue
		else:
		  category = str(user_input)
		  break

	#Extracting Prices and Benchmarks from SQL
	prices = pd.read_sql_table(category, con = engine,
	                                 schema = 'bloomberg_prices')
	prices.index=np.array(prices['row_names'], dtype='datetime64')
	prices = prices.iloc[:,1:]
	prices = prices.loc[~prices.index.duplicated(keep='first')]

	try:
		benchmarks = pd.read_sql_table('benchmark_prices', con = engine,
		                                   schema = 'bloomberg_prices')
		benchmarks.index=np.array(benchmarks['row_names'], dtype='datetime64')
		benchmarks = benchmarks.loc[~benchmarks.index.duplicated(keep='first')]
		benchmarks = benchmarks.iloc[:,1:]

		benchmark_tickers = pd.read_sql_table('benchmark_tickers', con = engine,
		                                   schema = 'bloomberg_prices')
		index = benchmarks[benchmark_tickers[category][0].replace(" ",".")]
		prices = pd.concat([pd.DataFrame(index),prices],axis=1)

	except KeyError:
		pass

	prices = prices.fillna(method='ffill')

	def to_returns(prices):
	    return prices.iloc[1:].values / prices.iloc[0:-1] - 1

	def calc_max_drawdown(prices):
	    return (prices / prices.expanding(min_periods=1).max()).min() - 1

	def oneyear_calc(df):

	    def downmarket_capture(price):
	        market = to_returns(df.iloc[:,0])
	        market = market[market < 0]

	        returns = to_returns(price)
	        returns = returns[returns.index.isin(market.index) == True]

	        market_cagr = (market + 1).product()
	        returns_cagr = (returns + 1).product()

	        return returns_cagr/market_cagr

	    def upmarket_capture(price):
	        market = to_returns(df.iloc[:,0])
	        market = market[market > 0]

	        returns = to_returns(price)
	        returns = returns[returns.index.isin(market.index) == True]

	        market_cagr = (market + 1).product()
	        returns_cagr = (returns + 1).product()

	        return returns_cagr/market_cagr

	    def period_returns(price):
	        return (price.iloc[-1]/price.iloc[0])**(1/1)-1

	    new_data['Returns_1yr']= df.apply(period_returns)
	    new_data['PositiveSD_1yr']=df.apply(to_returns)[df.apply(to_returns) > 0].apply(np.std) * np.sqrt(252)
	    new_data['NegativeSD_1yr']=df.apply(to_returns)[df.apply(to_returns) < 0].apply(np.std) * np.sqrt(252)
	    new_data['Max_Drawdown_1yr']=df.apply(calc_max_drawdown)
	    new_data['Upside_Capture_1yr'] = df.apply(upmarket_capture)
	    new_data['Downside_Capture_1yr'] = df.apply(downmarket_capture)

	def threeyear_calc(df):

	    def downmarket_capture(price):
	        market = to_returns(df.iloc[:,0])
	        market = market[market < 0]

	        returns = to_returns(price)
	        returns = returns[returns.index.isin(market.index) == True]

	        market_cagr = (market + 1).product()
	        returns_cagr = (returns + 1).product()

	        return returns_cagr/market_cagr

	    def upmarket_capture(price):
	        market = to_returns(df.iloc[:,0])
	        market = market[market > 0]

	        returns = to_returns(price)
	        returns = returns[returns.index.isin(market.index) == True]

	        market_cagr = (market + 1).product()
	        returns_cagr = (returns + 1).product()

	        return returns_cagr/market_cagr

	    def period_returns(price):
	        return (price.iloc[-1]/price.iloc[0])**(1/3)-1	    

	    new_data['Returns_3yr']= df.apply(period_returns)
	    new_data['PositiveSD_3yr']=df.apply(to_returns)[df.apply(to_returns) > 0].apply(np.std) * np.sqrt(252)
	    new_data['NegativeSD_3yr']=df.apply(to_returns)[df.apply(to_returns) < 0].apply(np.std) * np.sqrt(252)
	    new_data['Max_Drawdown_3yr']=df.apply(calc_max_drawdown)
	    new_data['Upside_Capture_3yr'] = df.apply(upmarket_capture)
	    new_data['Downside_Capture_3yr'] = df.apply(downmarket_capture)

	def fiveyear_calc(df):

	    def downmarket_capture(price):
	        market = to_returns(df.iloc[:,0])
	        market = market[market < 0]

	        returns = to_returns(price)
	        returns = returns[returns.index.isin(market.index) == True]

	        market_cagr = (market + 1).product()
	        returns_cagr = (returns + 1).product()

	        return returns_cagr/market_cagr

	    def upmarket_capture(price):
	        market = to_returns(df.iloc[:,0])
	        market = market[market > 0]

	        returns = to_returns(price)
	        returns = returns[returns.index.isin(market.index) == True]

	        market_cagr = (market + 1).product()
	        returns_cagr = (returns + 1).product()

	        return returns_cagr/market_cagr

	    def period_returns(price):
	        return (price.iloc[-1]/price.iloc[0])**(1/5)-1

	    new_data['Returns_5yr']= df.apply(period_returns)
	    new_data['PositiveSD_5yr']=df.apply(to_returns)[df.apply(to_returns) > 0].apply(np.std) * np.sqrt(252)
	    new_data['NegativeSD_5yr']=df.apply(to_returns)[df.apply(to_returns) < 0].apply(np.std) * np.sqrt(252)
	    new_data['Max_Drawdown_5yr']=df.apply(calc_max_drawdown)
	    new_data['Upside_Capture_5yr'] = df.apply(upmarket_capture)
	    new_data['Downside_Capture_5yr'] = df.apply(downmarket_capture)

	def scoring_days(data):
	    matrix = []
	    if 'NegativeSD' in str(data.index[0]) or 'Downside' in str(data.index[0]):
	        for i in range(len(data.index)):
	            matrix.append(np.where(data.iloc[i,:] > 
	                                     data.iloc[i,:].quantile(1-percent),1,0))
	        np.matrix(matrix)
	        result = np.sum(matrix,axis=0)

	        return allfunds[data.index[0]][np.argwhere(result == np.percentile(result, 
	                90, interpolation ='nearest'))[0,0]-1]
	    else:
	        for i in range(len(data.index)):
	            matrix.append(np.where(data.iloc[i,:] > 
	                                     data.iloc[i,:].quantile(percent),1,0))
	        np.matrix(matrix)
	        result = np.sum(matrix,axis=0)

	        return allfunds[data.index[0]][np.argwhere(result == np.percentile(result, 
	                10, interpolation ='nearest'))[0,0]-1]


	#Creating a truth value for each fund based on defined criteria
	def parse_returns(row): 
		try:
		    if(
		       #1 Year Criteria
		      (row.Returns_1yr >= scoring_days(returns_1yr)) & 
		      (row.NegativeSD_1yr < scoring_days(negativeSD_1yr)) &
		      (row.Upside_Capture_1yr > scoring_days(Upside_1yr)) &
		      (row.Downside_Capture_1yr < scoring_days(Downside_1yr)) &

		       #3 Year Criteria
		      (row.Returns_3yr >= scoring_days(returns_3yr)) & 
		      (row.PositiveSD_3yr > scoring_days(positiveSD_3yr)) &
		      (row.NegativeSD_3yr < scoring_days(negativeSD_3yr)) &
		      (row.Upside_Capture_3yr > scoring_days(Upside_3yr)) &
		      (row.Downside_Capture_3yr < scoring_days(Downside_3yr)) &

		       #5 Year Criteria
		      (row.Max_Drawdown_5yr > scoring_days(MDD_5yr)) &
		      (row.Upside_Capture_5yr > scoring_days(Upside_5yr)) &
		      (row.Downside_Capture_5yr < scoring_days(Downside_5yr)) 
		       ):
		        return 1
		    else:
		        return 0
		except IndexError:
		    return 1


	def create_score(fund_data):    
	    Y = [0.0]*len(fund_data)
	    for i in range(len(fund_data)):
	        Y[i] = parse_returns(fund_data.ix[i])
	  
	    return Y

	def evaluate(model, test_features, test_labels):
		predictions = model.predict(test_features)
		accuracy = f1_score(predictions, test_labels, average='weighted')
		print('Model Performance')
		print('Accuracy = {:0.2f}'.format(accuracy))

		return accuracy

	def downmarket_capture_one(price):
	    market = to_returns(oneyear.iloc[:,0])
	    market = market[market < 0]
	        
	    returns = to_returns(price)
	    returns = returns[returns.index.isin(market.index) == True]
	          
	    market_cagr = (market + 1).product()
	    returns_cagr = (returns + 1).product()
	     
	    return returns_cagr/market_cagr

	def upmarket_capture_one(price):
	    market = to_returns(oneyear.iloc[:,0])
	    market = market[market > 0]
	        
	    returns = to_returns(price)
	    returns = returns[returns.index.isin(market.index) == True]
	          
	    market_cagr = (market + 1).product()
	    returns_cagr = (returns + 1).product()

	    return returns_cagr/market_cagr

	def downmarket_capture_three(price):
	    market = to_returns(threeyear.iloc[:,0])
	    market = market[market < 0]
	        
	    returns = to_returns(price)
	    returns = returns[returns.index.isin(market.index) == True]
	          
	    market_cagr = (market + 1).product()
	    returns_cagr = (returns + 1).product()
	     
	    return returns_cagr/market_cagr

	def upmarket_capture_three(price):
	    market = to_returns(threeyear.iloc[:,0])
	    market = market[market > 0]
	        
	    returns = to_returns(price)
	    returns = returns[returns.index.isin(market.index) == True]
	          
	    market_cagr = (market + 1).product()
	    returns_cagr = (returns + 1).product()
	     
	    return returns_cagr/market_cagr

	def downmarket_capture_five(price):
	    market = to_returns(fiveyear.iloc[:,0])
	    market = market[market < 0]
	        
	    returns = to_returns(price)
	    returns = returns[returns.index.isin(market.index) == True]
	          
	    market_cagr = (market + 1).product()
	    returns_cagr = (returns + 1).product()
	     
	    return returns_cagr/market_cagr

	def upmarket_capture_five(price):
	    market = to_returns(fiveyear.iloc[:,0])
	    market = market[market > 0]
	        
	    returns = to_returns(price)
	    returns = returns[returns.index.isin(market.index) == True]
	          
	    market_cagr = (market + 1).product()
	    returns_cagr = (returns + 1).product()
	     
	    return returns_cagr/market_cagr

	global backtesting
	global nav
	backtesting = pd.DataFrame()
	features = pd.DataFrame()
	nav = pd.DataFrame()

	engine = create_engine('mysql://root:0000@localhost/portfolio_values')
	#Asking User for input of Date
	#Model projects one year forward from this date
	date = input("Enter starting date to backtest(e.g. 2017-03-02 (YYYY-MM-DD): ")

	#Defining Horizon for testing (250 days)
	start = int(prices.index.get_loc(date, method='ffill'))
	end = int(start + 250)

	engine = create_engine('mysql://root:0000@localhost/bloomberg_prices')
	parameters = pd.read_sql_table('parameter_selection', con = engine,
	                               schema = 'bloomberg_prices')

	engine = create_engine('mysql://root:0000@localhost/portfolio_values')
	scoringtable = pd.read_sql_table('scoring_criteria', con = engine, schema = 'portfolio_values')
	percent = pd.DataFrame(scoringtable)['Percentile'][0]
	threshold = pd.DataFrame(scoringtable)['Fund_Scoring_Threshold'][0]

	for day in range(start,end,5):
	    prices = prices.dropna(axis=1, how='all')
	    oneyear = prices.ix[prices.index[day-250]:prices.index[day]]
	    threeyear = prices.ix[prices.index[day-750]:prices.index[day]]
	    fiveyear = prices.ix[prices.index[day-1250]:prices.index[day]]

	    new_data = pd.DataFrame()
	    oneyear_calc(oneyear)
	    threeyear_calc(threeyear)
	    fiveyear_calc(fiveyear)
	    new_data = new_data[parameters['Selected_Parameters'].values[parameters['Selected_Parameters'].values != np.array(None)]]
	    allfunds = new_data.drop(new_data.index[0])
	    new_set = allfunds.dropna()

	    returns_1yr = pd.DataFrame(columns = prices.columns)
	    negativeSD_1yr = pd.DataFrame(columns = prices.columns)
	    Upside_1yr = pd.DataFrame(columns = prices.columns)
	    Downside_1yr = pd.DataFrame(columns = prices.columns)

	    returns_3yr = pd.DataFrame(columns = prices.columns)
	    positiveSD_3yr = pd.DataFrame(columns = prices.columns)
	    negativeSD_3yr = pd.DataFrame(columns = prices.columns)
	    Upside_3yr = pd.DataFrame(columns = prices.columns)
	    Downside_3yr = pd.DataFrame(columns = prices.columns)

	    MDD_5yr = pd.DataFrame(columns = prices.columns)
	    Upside_5yr = pd.DataFrame(columns = prices.columns)
	    Downside_5yr = pd.DataFrame(columns = prices.columns)


	    for i in range(250,0,-1):
	        one = prices.ix[prices.index[day-i-250]:prices.index[day-i]]
	        three = prices.ix[prices.index[day-i-750]:prices.index[day-i]]
	        five = prices.ix[prices.index[day-i-1250]:prices.index[day-i]]
	        new_data = pd.DataFrame()

	        oneyear_calc(one)
	        threeyear_calc(three)
	        fiveyear_calc(five)
	        new_data = new_data[parameters['Selected_Parameters'].values[parameters['Selected_Parameters'].values != np.array(None)]]

	        try:
	        	returns_1yr = returns_1yr.append(new_data['Returns_1yr'].transpose())
	        except KeyError:
	        	pass

	        try:	
	        	negativeSD_1yr = negativeSD_1yr.append(new_data['NegativeSD_1yr'].transpose())
	        except KeyError:
	        	pass

	        try:
	        	Upside_1yr = Upside_1yr.append(new_data['Upside_Capture_1yr'].transpose())
	        except KeyError:
	        	pass

	        try:
	        	Downside_1yr = Downside_1yr.append(new_data['Downside_Capture_1yr'].transpose())
	        except KeyError:
	        	pass

	        try:	
	        	returns_3yr = returns_3yr.append(new_data['Returns_3yr'].transpose())
	        except KeyError:
	        	pass

	        try:	
	        	positiveSD_3yr = positiveSD_3yr.append(new_data['PositiveSD_3yr'].transpose())
	        except KeyError:
	        	pass

	        try:
	        	negativeSD_3yr = negativeSD_3yr.append(new_data['NegativeSD_3yr'].transpose())
	        except KeyError:
	        	pass

	        try:
	        	Upside_3yr = Upside_3yr.append(new_data['Upside_Capture_3yr'].transpose())
	        except KeyError:
	        	pass

	        try:
	        	Downside_3yr = Downside_3yr.append(new_data['Downside_Capture_3yr'].transpose())
	        except KeyError:
	        	pass

	        try:
	        	MDD_5yr = MDD_5yr.append(new_data['Max_Drawdown_5yr'].transpose())
	        except KeyError:
	        	pass

	        try:
	        	Upside_5yr = Upside_5yr.append(new_data['Upside_Capture_5yr'].transpose())
	        except KeyError:
	        	pass
	        try:
	        	Downside_5yr = Downside_5yr.append(new_data['Downside_Capture_5yr'].transpose())
	        except KeyError:
	        	pass

	    Y = create_score(new_set)

	    X = new_set.as_matrix()
	    Y = np.array(Y)

	    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10,bootstrap=False)
	    clf = clf.fit(X,Y)

	    #Tuning hyperparameters for overall model
	    train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.15, random_state = 42)

	    n_estimators = [int(x) for x in np.linspace(start = 1, stop = 500, num = 500)]
	    max_features = ['auto']
	    max_depth = [int(x) for x in np.linspace(1, 30, num = 30)]
	    max_depth.append(None)
	    min_samples_split = [2, 5]
	    min_samples_leaf = [1, 2, 4]
	    bootstrap = [True, False]

	    # Create the random grid to find optimal hyperparameters for our function
	    random_grid = {'n_estimators': n_estimators,
	                   'max_features': max_features,
	                   'max_depth': max_depth,
	                 'min_samples_split': min_samples_split,
	                 'min_samples_leaf': min_samples_leaf,
	                 'bootstrap': bootstrap}

	    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
	                                  n_iter = 10, scoring='f1', 
	                                  cv = 10, verbose=2, random_state=42, n_jobs=-1,
	                                  return_train_score=True)

	    # Fit the random search model
	    rf_random.fit(train_features, train_labels)
	    rf_random.best_params_

	    #Checking for improvement of tuned RF model over initial reference model created
	    def evaluate(model, test_features, test_labels):
	        predictions = model.predict(test_features)
	        accuracy = f1_score(predictions, test_labels, average='weighted')
	        print('Model Performance')
	        print('Accuracy = {:0.2f}'.format(accuracy))

	        return accuracy

	    base_accuracy = evaluate(clf, test_features, test_labels)
	    best_random = rf_random.best_estimator_
	    random_accuracy = evaluate(best_random, test_features, test_labels)

	    #print(('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy)))
	    feature_imp = best_random.feature_importances_
	    if all(feature == 0 for feature in feature_imp) == True:
	    	feature_imp = features.iloc[-1,:]
	    features = features.append(pd.DataFrame(feature_imp).transpose())

	    fund_adjust = new_set.copy()
	    fund_adjust['NegativeSD_3yr'] = 1 / fund_adjust['NegativeSD_3yr']
	    fund_adjust['NegativeSD_1yr'] = 1 / fund_adjust['NegativeSD_1yr']
	    fund_adjust['Downside_Capture_3yr'] = 1 / fund_adjust['Downside_Capture_3yr']
	    fund_adjust['Downside_Capture_1yr'] = 1 / fund_adjust['Downside_Capture_1yr']
	    fund_adjust['Downside_Capture_5yr'] = 1 / fund_adjust['Downside_Capture_5yr']

	    #Calculating the percentile of each fund's parameter across the fund universe
	    def scoring(x):
	        i,j = fund_adjust.shape
	        total_scores = np.zeros(shape=(i,))
	        for m in range(0,i):
	            percentile = np.zeros(shape=(j,))
	            for n in range(0, j):
	                percentile[n]=scipy.stats.percentileofscore(x.iloc[:,n],x.iloc[m,n])/100
	            total_scores[m]=sum(percentile*feature_imp)
	        return total_scores

	    #Calculating the final score by taking feature importance score multiplied by the percentile of each parameter value
	    N = fund_adjust.loc[:, fund_adjust.columns != 'Predicted_result']
	    new_set["Final_Score"] = scoring(N)
	    best_funds = new_set.sort_values(['Final_Score'],
	                                    ascending=False)
	    backtesting = backtesting.append(pd.DataFrame(best_funds.index[0:10]).transpose())

	    best_ten_funds = best_funds[0:10]
	    #10 Choose 5
	    portfolio_list = list(combinations(best_ten_funds.index, 5))
	    portfolios = np.asarray(portfolio_list)

	    i = portfolios.shape[0]
	    j = portfolios.shape[1]

	    #Creating a synthetic NAV by equally weighting all funds in the portfolio
	    synthetic_nav = np.zeros((i,fiveyear.shape[0]))

	    for x in range(i):

	        units = np.array(1/fiveyear.iloc[:,np.where(np.isin(fiveyear.columns, portfolios[x]))[0].tolist()] * 1/5)[0]
	        synthetic_nav[x] = (units * fiveyear.iloc[:,np.where(np.isin(fiveyear.columns, portfolios[x]))[0].tolist()]).sum(axis=1)

	    synthetic_nav = pd.DataFrame(np.transpose(synthetic_nav))

	    synthetic_nav.index = fiveyear.index
	    daily_returns = fiveyear.dropna(axis = 1, how = 'all').pct_change()
	    returns_list = synthetic_nav.apply(to_returns)
	    returns_list.index = synthetic_nav.index[1:]

	    #Calculating all parameter values for each portfolio
	    returns_three =  list(round((synthetic_nav.iloc[-1,:]/synthetic_nav.ix[synthetic_nav.index[-750] ,:]**(1/3) - 1),4))
	    returns_one =  list(round((synthetic_nav.iloc[-1,:]/synthetic_nav.ix[synthetic_nav.index[-250],:] - 1),4))

	    positive_SD_three = list(returns_list.ix[returns_list.index[-750]:][returns_list.ix[returns_list.index[-750]:] > 0].apply(np.std) * np.sqrt(252))
	    negative_SD_three = list(returns_list.ix[returns_list.index[-750]:][returns_list.ix[returns_list.index[-750]:] < 0].apply(np.std) * np.sqrt(252))
	    upmarket_three = list(synthetic_nav.apply(upmarket_capture_three))
	    downmarket_three = list(synthetic_nav.apply(downmarket_capture_three))

	    negative_SD_one = list(returns_list.ix[returns_list.index[-250]:][returns_list.ix[returns_list.index[-250]:] < 0].apply(np.std) * np.sqrt(252))
	    upmarket_one = list(synthetic_nav.apply(upmarket_capture_one))
	    downmarket_one = list(synthetic_nav.apply(downmarket_capture_one))

	    mdd_five = list(synthetic_nav.apply(calc_max_drawdown))
	    upmarket_five = list(synthetic_nav.apply(upmarket_capture_five))
	    downmarket_five = list(synthetic_nav.apply(downmarket_capture_five))

	    values = np.stack((returns_one,
	             negative_SD_one, upmarket_one, downmarket_one,
	                       returns_three, positive_SD_three, negative_SD_three,
	             upmarket_three, downmarket_three, mdd_five, upmarket_five, downmarket_five), axis=-1)

	    final_5_funds = pd.DataFrame(np.concatenate([portfolios, values], axis = 1))
	    final_5_funds.columns = ['Fund 1', 'Fund 2', 'Fund 3','Fund 4','Fund 5'] + list(parameters['Selected_Parameters'].values[parameters['Selected_Parameters'].values != np.array(None)])

	    final_portfolios = final_5_funds
	    final_portfolios.iloc[:,5:] = final_portfolios.iloc[:,5:].astype(dtype='float')
	    #Adjusting for scores
	    scores = final_portfolios.copy()

	    scores['NegativeSD_3yr'] = 1 / scores['NegativeSD_3yr']
	    scores['NegativeSD_1yr'] = 1 / scores['NegativeSD_1yr']
	    scores['Downside_Capture_3yr'] = 1 / scores['Downside_Capture_3yr']
	    scores['Downside_Capture_1yr'] = 1 / scores['Downside_Capture_1yr']

	    i, j = final_portfolios.iloc[:,5:].shape
	    total_scores = np.zeros(shape=(i,))

	    #Multiplying original feature importance score by the percentile of each portfolio's parameter value across entire universe
	    for m in range(0,i):
	        percentile = np.zeros(shape=(j,))
	        for n in range(5, (j + 5)):
	            percentile[(n - 5)]=scipy.stats.percentileofscore(scores.iloc[:,n],scores.iloc[m,n])/100
	            total_scores[m]=sum(percentile*feature_imp)

	    final_portfolios['Score'] = total_scores
	    final_portfolios.sort_values(['Score'], ascending =False)

	    fund_combinations = final_portfolios.iloc[:,0:5]

	    if day == int(prices.index.get_loc(date)):
	        myportfolio = set(final_portfolios.iloc[0,0:5].values)
	        sharecount = 100/5/prices.ix[day,prices.columns.isin(myportfolio)]

	    diff_comb = []
	    for x in range(len(fund_combinations.index)):
	        diff_comb.append(set(fund_combinations.iloc[x,:]))

	    if myportfolio in diff_comb:
	        i = diff_comb.index(myportfolio)
	        difference = final_portfolios.iloc[0,-1] - final_portfolios.iloc[i,-1]
	        if difference > (threshold * final_portfolios.iloc[i,-1]):
	            out = list(myportfolio- set(final_portfolios.iloc[0,0:5]))
	            _in = list(set(final_portfolios.iloc[0,0:5]) - myportfolio)
	            addition = (sharecount.loc[out]*prices.ix[day,prices.columns.isin(out)].reindex(out).values)/prices.ix[day,prices.columns.isin(_in)].reindex(_in).values
	            addition.index = _in
	            sharecount = sharecount.drop(out, axis = 0)
	            sharecount = sharecount.append(addition)
	            myportfolio = set(sharecount.index)

	    elif myportfolio not in diff_comb:
	        out = list(myportfolio- set(final_portfolios.iloc[0,0:5]))
	        _in = list(set(final_portfolios.iloc[0,0:5]) - myportfolio)
	        addition = (sharecount.loc[out]*prices.ix[day,prices.columns.isin(out)].reindex(out).values)/prices.ix[day,prices.columns.isin(_in)].reindex(_in).values
	        addition.index = _in
	        sharecount = sharecount.drop(out, axis = 0)
	        sharecount = sharecount.append(addition)
	        myportfolio = set(sharecount.index)	

	    print(myportfolio)            
	    fund_values = list(sharecount.index) + [(sharecount * prices.iloc[day,:]).T.sum()]
	    nav = nav.append(pd.DataFrame(fund_values).T, ignore_index = True)
	    print(nav)

	writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')
	pd.DataFrame(features, columns = list(parameters['Selected_Parameters'].values[parameters['Selected_Parameters'].values != np.array(None)])).to_excel(writer, sheet_name='Feature Importance')
	backtesting.to_excel(writer, sheet_name='Top Portfolios')
	nav.to_excel(writer, sheet_name='NAV', columns = ['Fund 1', 'Fund 2', 'Fund 3', 'Fund 4', 'Fund 5', 'NAV'])

