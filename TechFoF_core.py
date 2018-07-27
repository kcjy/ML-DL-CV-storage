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
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, r2_score
from scipy.stats import norm
from collections import defaultdict
from sqlalchemy import create_engine
import os
from sqlalchemy.orm import sessionmaker, scoped_session

def to_returns(prices):
      return prices.iloc[1:].values / prices.iloc[0:-1] - 1

def calc_max_drawdown(prices):
    return (prices / prices.expanding(min_periods=1).max()).min() - 1

#Calculating parameters for 1 year timeframe within chosen 6 parameters
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

#Calculating parameters for 3 year timeframe within chosen 6 parameters
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

#Calculating parameters for 5 year timeframe within chosen 6 parameters
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

#Functions to be used at Portfolio calculations level
#Most are repeated from fund-level calculations with only minor changes madde to timeframes
#Separate functions are created to allow pandas apply function to be run on dataframe

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

#Determining the number of days which a fund is above the 16th percentile (Median - 1 SD)for each factor
#over a period of 250 days
#NegativeSD and Downside Capture have inverse numerical properties, where smaller values are better
#Normal distribution assumption is made, and likely to hold true based on frequency distribution tests done previously

def scoring_days(data):
  matrix = []
  if 'NegativeSD' in str(data.index[0]) or 'Downside' in str(data.index[0]):
    for i in range(len(data.index)):
        matrix.append(np.where(data.iloc[i,:] > 
                                 data.iloc[i,:].quantile(1-percent),1,0))
    np.matrix(matrix)
    result = np.sum(matrix,axis=0)
    
    return allfunds[data.index[0]][np.argwhere(result == np.percentile(result, 
            85, interpolation ='nearest'))[0,0]-1]
  else:
    for i in range(len(data.index)):
        matrix.append(np.where(data.iloc[i,:] > 
                                 data.iloc[i,:].quantile(percent),1,0))
    np.matrix(matrix)
    result = np.sum(matrix,axis=0)
    
    return allfunds[data.index[0]][np.argwhere(result == np.percentile(result, 
            15, interpolation ='nearest'))[0,0]-1]


#Creating a truth value for each fund based on defined criteria
#To identify good funds vs bad funds on any given day
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

#Application of parse_returns function to create array Y
#1 for good funds, 0 for bad funds  
def create_score(fund_data):    
    Y = [0.0]*len(fund_data)
    for i in range(len(fund_data)):
        Y[i] = parse_returns(fund_data.ix[i])
    
    return Y

#Function to evaluate model accuracy
def evaluate(model, test_features, test_labels):
  predictions = model.predict(test_features)
  accuracy = f1_score(predictions, test_labels, average='weighted')
  print('Model Performance')
  print('Accuracy = {:0.2f}'.format(accuracy))

  return accuracy

if __name__ == '__main__':
  category = 'core'

  #Download data from MySQL database
  engine = create_engine('mysql://root:0000@localhost/bloomberg_prices')
  prices = pd.read_sql_table(category, con = engine,
                                     schema = 'bloomberg_prices')
  prices.index=np.array(prices['row_names'], dtype='datetime64')
  prices = prices.iloc[:,1:]

  try:
    benchmarks = pd.read_sql_table('benchmark_prices', con = engine,
                                       schema = 'bloomberg_prices')
    benchmarks.index=np.array(benchmarks['row_names'], dtype='datetime64')
    benchmarks = benchmarks.iloc[:,1:]

    benchmark_tickers = pd.read_sql_table('benchmark_tickers', con = engine,
                                       schema = 'bloomberg_prices')
    index = benchmarks[benchmark_tickers[category][0].replace(" ",".")]

    prices = pd.concat([pd.DataFrame(index),prices],axis=1)
  except KeyError:
    pass

  prices = prices.fillna(method='ffill')

  #Defintion of functions to be used for parameter calculation

  #Creating subsets of 1,3 & 5 Year data
  oneyear = prices.ix[prices.index[-250]:]
  threeyear = prices.ix[prices.index[-750]:]
  fiveyear = prices.ix[prices.index[-1250]:]

  #Calculating all parameters for 1,3 & 5 year data
  new_data = pd.DataFrame()
  oneyear_calc(oneyear)
  threeyear_calc(threeyear)
  fiveyear_calc(fiveyear)

  engine = create_engine('mysql://root:0000@localhost/bloomberg_prices')
  parameters = pd.read_sql_table('parameter_selection', con = engine,
                                   schema = 'bloomberg_prices')
  new_data = new_data[parameters['Selected_Parameters'].values[parameters['Selected_Parameters'].values != np.array(None)]]

  allfunds = new_data.drop(new_data.index[0])

  engine = create_engine('mysql://root:0000@localhost/portfolio_values')
  scoringtable = pd.read_sql_table('scoring_criteria', con = engine,
                                   schema = 'portfolio_values')
  percent = pd.DataFrame(scoringtable)['Percentile'][0]
  threshold = pd.DataFrame(scoringtable)['Fund_Scoring_Threshold'][0]

  #Creating dataframes to store rolling values
  #This is used to generate frequency distributions of each parameter used
  #Objective is to determine a numerical value for each parameter to define a good fund
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

  #Finding all parameters for all funds over past 1 year to formulate scoring criteria
  for i in range(250,0,-1):
      one = prices.ix[prices.index[-i-250]:prices.index[-i]]
      three = prices.ix[prices.index[-i-750]:prices.index[-i]]
      five = prices.ix[prices.index[-i-1250]:prices.index[-i]]
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

  
  new_set = allfunds.dropna()
  Y = create_score(new_set)

  X = new_set.as_matrix()
  Y = np.array(Y)

  #Training Reference Model
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

  #Random Search to identify optimal hyperparameters
  #Using reference model named clf
  rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                                n_iter = 10, scoring='f1', 
                                cv = 10, verbose=2, random_state=42, n_jobs=-1,
                                return_train_score=True)
      
  # Fit the random search model
  rf_random.fit(train_features, train_labels)
  rf_random.best_params_

  #Adjusting for parameters where smaller numerical values are better for a fund
  base_accuracy = evaluate(clf, test_features, test_labels)
  best_random = rf_random.best_estimator_
  random_accuracy = evaluate(best_random, test_features, test_labels)

  #print(('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy)))
  feature_imp = best_random.feature_importances_
  print(feature_imp)

  fund_adjust = new_set.copy()
  try:
    fund_adjust['NegativeSD_3yr'] = 1 / fund_adjust['NegativeSD_3yr']
    fund_adjust['NegativeSD_1yr'] = 1 / fund_adjust['NegativeSD_1yr']
    fund_adjust['Downside_Capture_3yr'] = 1 / fund_adjust['Downside_Capture_3yr']
    fund_adjust['Downside_Capture_1yr'] = 1 / fund_adjust['Downside_Capture_1yr']
    fund_adjust['Downside_Capture_5yr'] = 1 / fund_adjust['Downside_Capture_5yr']
  except KeyError:
    pass

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


  #Portfolio Level calculations
  #To identify best combinations of funds to form best portfolio
  #Most of the calculations will be a replica from the fund-level calculations above

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
  #portfolios = pd.DataFrame(portfolios)

  synthetic_nav.index = fiveyear.index
  daily_returns = fiveyear.dropna(axis = 1, how = 'all').pct_change()
  returns_list = synthetic_nav.apply(to_returns)
  returns_list.index = synthetic_nav.index[1:]

  #Calculating all parameter values for each portfolio
  returns_three =  list(round((synthetic_nav.iloc[-1,:]/synthetic_nav.ix[synthetic_nav.index[-750] ,:]**(1/3) - 1),4))
  returns_one =  list(round((synthetic_nav.iloc[-1,:]/synthetic_nav.ix[synthetic_nav.index[-250],:] - 1),4))
  #returns_five =  list(round((synthetic_nav.iloc[-1,:]/synthetic_nav.ix[synthetic_nav.index[-1250],:]**(1/5) - 1),4))

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

  try:
    scores['NegativeSD_3yr'] = 1 / scores['NegativeSD_3yr']
    scores['NegativeSD_1yr'] = 1 / scores['NegativeSD_1yr']
    scores['Downside_Capture_3yr'] = 1 / scores['Downside_Capture_3yr']
    scores['Downside_Capture_1yr'] = 1 / scores['Downside_Capture_1yr']
  except KeyError:
    pass

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

  fund_combinations = final_portfolios.iloc[:, 0:5]

  #Retrieving current portfolio stored in MySQL database
  #Used to compare our current portfolio with top portfolio, and decide whether portfolio should be changed

  engine = create_engine('mysql://root:0000@localhost/portfolio_values')
  current_portfolio = pd.read_sql_table('core_portfolio', con = engine,
                                     schema = 'portfolio_values').iloc[0:5,1:]

  #Determining whether fund should be changed.
  #If changed, current portfolio will be updated in MySQL

  for i in range(len(fund_combinations.index)):
    my_list = list(fund_combinations.iloc[i,:])
    my_list = [x for x in my_list if not isinstance(x, float)]
    if set(my_list) == set(list(zip(*current_portfolio.values.tolist()))[0]):
        difference = final_portfolios.iloc[0,-1] - final_portfolios.iloc[i,-1]
        #3 Percent Threshold
        if difference > (threshold * final_portfolios.iloc[i,-1]):
            current_portfolio = set(final_portfolios.iloc[0,0:5])
            current_portfolio.to_sql('core_portfolio', con=engine,if_exists = 'replace')
            print(current_portfolio)
            break
        else:
            print(current_portfolio)

  #Storing top portfolios in MySQL database for future use
  #Appending to existing MySQL table to store top portfolios
  pd.DataFrame(final_portfolios.iloc[0,:].values,index = list(pd.read_sql_table('top_core_portfolios', con = engine,
                schema = 'portfolio_values').columns[1:]), columns = [prices.index[-1]]).transpose().to_sql('top_core_portfolios', con=engine,if_exists = 'append')
  pd.DataFrame(feature_imp, index = list(pd.read_sql_table('core_features', con = engine,
                schema = 'portfolio_values').columns[1:]), columns = [prices.index[-1]]).transpose().to_sql('core_features', con=engine,if_exists = 'append')