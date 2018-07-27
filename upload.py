import pandas as pd
from sqlalchemy import create_engine
import xlwings as xw

if __name__ == '__main__':
	#Locating excel file
	xl = xw.Book('C:/Users/Admin/Desktop/Work Folder/Code/Executables (Exe)/Fund of Funds Interface.xlsm')

	#Identifying relevant sheet names that need to be updated
	sheet_names = []
	for i in xl.sheets:
		sheet_names.append(i)
	sheet_names = sheet_names[1:-3]

	#The following lines extract the necessary values from each worksheet from the Excel file
	#Data values are specifically identified for each worksheet
	engine = create_engine('mysql://root:0000@localhost/bloomberg_prices')
	sheet = xl.sheets[sheet_names[0]]
	bdp_bdh = sheet['A1:B2'].options(pd.DataFrame, index=False, header=True).value
	bdp_bdh.to_sql('bdp_bdh', con=engine,if_exists = 'replace', index = False)

	sheet = xl.sheets[sheet_names[1]]
	benchmark_tickers = sheet['A1:D2'].options(pd.DataFrame, index=False, header=True).value
	benchmark_tickers.to_sql('benchmark_tickers', con=engine,if_exists = 'replace', index = False)

	sheet = xl.sheets[sheet_names[2]]
	fund_tickers = sheet['A1:E56'].options(pd.DataFrame, index=False, header=True).value
	fund_tickers.to_sql('fund_tickers', con=engine,if_exists = 'replace', index = False)

	sheet = xl.sheets[sheet_names[3]]
	parameter_selection = sheet['A1:C13'].options(pd.DataFrame, index=False, header=True).value
	parameter_selection.to_sql('parameter_selection', con=engine,if_exists = 'replace', index = False)


	engine = create_engine('mysql://root:0000@localhost/portfolio_values')

	sheet = xl.sheets[sheet_names[4]]
	core_portfolio = sheet['A1:B6'].options(pd.DataFrame, index=False, header=True).value
	core_portfolio.to_sql('core_portfolio', con=engine,if_exists = 'replace', index = False)

	sheet = xl.sheets[sheet_names[5]]
	satellite_portfolio = sheet['A1:B6'].options(pd.DataFrame, index=False, header=True).value
	satellite_portfolio.to_sql('satellite_portfolio', con=engine,if_exists = 'replace', index = False)

	sheet = xl.sheets[sheet_names[6]]
	indiafof_portfolio = sheet['A1:B6'].options(pd.DataFrame, index=False, header=True).value
	indiafof_portfolio.to_sql('indiafof_portfolio', con=engine,if_exists = 'replace', index = False)

	sheet = xl.sheets[sheet_names[7]]
	scoring_criteria = sheet['A1:C2'].options(pd.DataFrame, index=False, header=True).value
	scoring_criteria.to_sql('scoring_criteria', con=engine,if_exists = 'replace', index = False)
