##Created by Kenneth
#Model aims to identify under/over-performing bonds based on regression of ZSpread with Market
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pylab
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates


#User Inputs for required fields
#Input data should be Historical ZSpread data from Bloomberg
#Inputs are file locations to allow customizability for users
filename = pd.ExcelFile('zSpread.xlsx')
graphs = "C:/Users/Silverdale13/Documents/ZSpread/Graphs/"
benchmark = input("Name of Benchmark (Column Name of Benchmark Index): ")

data = pd.DataFrame(filename.parse('DATA'))

#Assumption: 1 Year = 250 days
#This is needed as allowing user to input date may result in a non-trading day
#Therefore, we make this assumption to ensure model is robust
oneyear = data[data.index[-250]:]
sixmonth = data[data.index[-120]:]
threemonth = data[data.index[-60]:]

result = pd.DataFrame()
slope = pd.DataFrame()
stdev = pd.DataFrame()

class chart_results:

  def plot_graph(self):
        namelist.append(self.column)
        valuelist.append(((self.regr_res.iloc[-1,1]-self.regr_res.iloc[-1,2])/
                         np.std(self.regr_res['Actual'])))
        fig, ax = plt.subplots()
        ax.plot(self.y.index, self.regr_res[benchmark], 
        self.y.index, self.regr_res['Actual'],
        self.y.index, self.regr_res['Predict'],
        self.y.index, self.regr_res['Lower'],
        self.y.index, self.regr_res['Upper'])
        plt.ylabel('Basis Points')
        plt.xlabel('Date',fontsize = 'xx-small')
        ax.legend(['Market', 'Actual', 'Predict', 'Lower', 'Upper'],
          loc='lower right',fontsize='xx-small')

        if self.period == '1year':
          myFmt = mdates.DateFormatter('%b-%Y')
          ax.xaxis.set_major_formatter(myFmt)
          plt.title(str(self.column) + '- 1 Year')
          plt.savefig(graphs + str(self.column) + '- 1 Year.png', dpi=300)
          #plt.show()

        elif self.period == '6mth':
          myFmt = mdates.DateFormatter('%b-%Y')
          ax.xaxis.set_major_formatter(myFmt)
          plt.title(str(self.column) + '- 6 Month')
          plt.savefig(graphs + str(self.column) + '- 6 Month.png', dpi=300)
          #plt.show()

        elif self.period == '3mth':
          myFmt = mdates.DateFormatter('%d-%b')
          ax.xaxis.set_major_formatter(myFmt)
          plt.title(str(self.column) + '- 3 Month')
          plt.savefig(graphs + str(self.column) + '- 3 Month.png', dpi=300)
          #plt.show()

  def regression(self, df, period = ['1year', '6mth', '3mth']):
      self.period = period
      # Remember to change market name
      df=df.dropna(axis=1, how="all")
      df = df.fillna(method='ffill')
      new_columns = [w.replace('/', '-') for w in df.columns]
      df.columns = new_columns
      Y = df.loc[:, df.columns != benchmark]
      X = df.loc[:,benchmark]
      X_const = sm.add_constant(X)
      
      global namelist 
      global valuelist
      
      namelist = []
      valuelist = []
      
      for column in Y:
          self.column = column
          self.y = Y[column]
          results = sm.regression.linear_model.OLS(self.y,X_const).fit()
          
          self.regr_res = pd.DataFrame()
          self.regr_res[benchmark] = X
          self.regr_res['Actual'] = self.y
          self.regr_res['Predict'] = results.params[1] * X + results.params[0]
          if results.params[1] <= 0.5:
            continue

          self.regr_res['Lower'] = self.regr_res['Predict'] - (1.5*np.std(self.regr_res['Actual']))
          self.regr_res['Upper'] = self.regr_res['Predict'] + (1.5*np.std(self.regr_res['Actual']))
          
          if self.regr_res.iloc[-1,1] < self.regr_res.iloc[-1,3]:
              if -15 < ((self.regr_res.iloc[-1,1]-self.regr_res.iloc[-1,
                            2])/np.std(self.regr_res['Actual'])) < 15:
                self.plot_graph()

          elif self.regr_res.iloc[-1,1] > self.regr_res.iloc[-1,4]:
              if -15 < ((self.regr_res.iloc[-1,1]-self.regr_res.iloc[-1,
                         2])/np.std(self.regr_res['Actual'])) < 15:
                self.plot_graph()

      namelist = pd.Series((v for v in namelist))
      valuelist = pd.Series((v for v in valuelist))

CR = chart_results()
CR.regression(oneyear, period='1year')
result = pd.concat([result, namelist, valuelist],ignore_index=True, axis=1)

CR.regression(sixmonth, period='6mth')
result = pd.concat([result, namelist, valuelist],ignore_index=True, axis=1)

CR.regression(threemonth, period='3mth')
result = pd.concat([result, namelist, valuelist],ignore_index=True, axis=1)

result.columns = ['OneYear',
                  'OneYear_SD',
                  'SixMonth',
                  'SixMonth_SD',
                  'ThreeMonth',
                  'ThreeMonth_SD']

first = result.iloc[:,0:2].sort_values('OneYear_SD')
second = result.iloc[:,2:4].sort_values('SixMonth_SD')
third = result.iloc[:,4:6].sort_values('ThreeMonth_SD')

final_results = pd.DataFrame()
final_results['OneYear'] = first['OneYear'].values
final_results['OneYear_SD'] = first['OneYear_SD'].values
final_results['SixMonth'] = second['SixMonth'].values
final_results['SixMonth_SD'] = second['SixMonth_SD'].values
final_results['ThreeMonth'] = third['ThreeMonth'].values
final_results['ThreeMonth_SD'] = third['ThreeMonth_SD'].values

final_results.to_csv(e'results.csv')
