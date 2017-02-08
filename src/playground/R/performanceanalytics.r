library(PerformanceAnalytics)

percentdiff <- function(x) {
  N <- length(x)
  xdiff <- numeric(N-1)
  
  for(i in 1:N-1) {
 			print(i);
			xdiff[i] = (x[i+1]/x[i]) - 1;
  }
  return(xdiff)
}

percentdiffShare <- function(x) {
  #print(x)
  N <- length(x)
  #print(N)
  
  xdiff <- numeric(N-1)
  #print(xdiff)
  
  for(i in 1:(N-1)) {
 			#print(i);
			#print(as.double(x[i]))
			#print(as.double(x[i+1]))
			xdiff[i] = (as.double(x[i+1])/as.double(x[i])) - 1;
			
  } 
  return(xdiff)
}

# general compute value at risk
computeVar <- function(x, openpos, price) {
  pct <- percentdiff(x)
  return(VaR(pct, p = 0.95)[1]*openpos*price)
}

#source("D:\\xampp\\htdocs\\gpu_freedom\\src\\playground\\R\\performanceanalytics.r")
#library(quantmod)
#getSymbols("TSLA",src="yahoo") # from google finance
#barChart(TSLA)
#require(TTR)
#chartSeries(TSLA)
#addMACD()
#addBBands()
#computeVarShare(TSLA,9)
computeVarShare <- function(x, nbShares) {
  N <- nrow(x)
  ndaysTable = x[(N-90):N, 4] # Close Price column is the 4th column
  curPrice <- as.double(x[N,4]) # latest available price
  
  pct <- percentdiffShare(ndaysTable)
  return (VaR(pct, p = 0.95)[1]*nbShares*curPrice) 
}

myPortfolioVar <- function() {
	getSymbols("ABBN.VX",src="yahoo")
	getSymbols("ALPH.SW",src="yahoo")
	getSymbols("CAT",src="yahoo")
	getSymbols("DDD",src="yahoo")
	getSymbols("DE",src="yahoo")
	getSymbols("DIS",src="yahoo")
	getSymbols("FMC",src="yahoo")
	getSymbols("GM",src="yahoo")
	getSymbols("GRMN",src="yahoo")
	getSymbols("GSK",src="yahoo")
	getSymbols("MA",src="yahoo")
	getSymbols("NESN.VX",src="yahoo")
	getSymbols("NOVN.VX",src="yahoo")
	getSymbols("NVDA",src="yahoo")
	getSymbols("ROG.VX",src="yahoo")
	getSymbols("SMTC",src="yahoo")
	getSymbols("TM",src="yahoo")
	getSymbols("TSLA",src="yahoo")
    getSymbols("TXN",src="yahoo")
	getSymbols("UBXN.SW",src="yahoo")
	getSymbols("XLNX",src="yahoo")
	
	print("VaR ABBN.VX:")
	print(computeVarShare(ABBN.VX,45))
	
	print("VaR ALPH.SW:")
	print(computeVarShare(ALPH.SW,2))
	
	print("VaR CAT:")
	print(computeVarShare(CAT,6))
	
	print("VaR DDD:")
	print(computeVarShare(DDD,50))
	
	print("VaR DE:")
	print(computeVarShare(DE,6))
	
	print("VaR DIS:")
	print(computeVarShare(DIS,1))
	
	print("VaR FMC:")
	print(computeVarShare(FMC,15))
	
	print("VaR GM:")
	print(computeVarShare(GM,19))
	
	print("VaR GRMN:")
	print(computeVarShare(GRMN,5))
	
	print("VaR GSK:")
	print(computeVarShare(GSK,16))
	
	print("VaR MA:")
	print(computeVarShare(MA,4))
	
	print("VaR NESN.VX:")
	print(computeVarShare(NESN.VX,12))
	
	print("VaR NOVN.VX:")
	print(computeVarShare(NOVN.VX,13))
	
	print("VaR NVDA:")
	print(computeVarShare(NVDA,33))
	
	print("VaR ROG.VX:")
	print(computeVarShare(ROG.VX,2))
	
	print("VaR SMTC:")
	print(computeVarShare(SMTC,16))
	
	print("VaR TM:")
	print(computeVarShare(TM,6))
	
	print("VaR TSLA:")
	print(computeVarShare(TSLA,9))

	print("VaR TXN:")
	print(computeVarShare(TXN,9))

	print("VaR UBXN.SW:")
	print(computeVarShare(UBXN.SW,4))

	print("VaR XLNX:")
	print(computeVarShare(XLNX,15))

}


# source("C:\\xampp\\htdocs\\gpu_freedom\\src\\playground\\R\\jdbc.r")
# source("C:\\xampp\\htdocs\\gpu_freedom\\src\\playground\\R\\performanceanalytics.r")
# computeVarBTC(0.3) # i currently own 0.3 BTC 
# it returns the amount of money i can loose on the next day 
# with 95% confidence level and 60 days time horizon in the past

# compute value at risk for bitcoins
#requires jdbc.r to be resourced
computeVarBTC <- function(btc_open) {
  print("last price (USD):")
  print(b_last_price$price)
  print("Value at Risk, 60 day interval, 95% confidence level (USD):")
  return(computeVar(b_avg_price_last_60$usd, btc_open, b_last_price$price))
}  

computeTotalVarBTC <- function() {

 print("short term open (BTC)")
 print(b_btc_shortterm_open$btc) 
 print("mid term open (BTC)")
 print(b_btc_midterm_open$btc)
 print("long term open (BTC)")
 print(b_btc_longterm_open$btc) 
 print("tiz open (BTC)")
 print(b_btc_tiz_open$btc)
 
 total = b_btc_shortterm_open$btc + b_btc_midterm_open$btc + b_btc_longterm_open$btc + b_btc_tiz_open$btc
 print ("total open (BTC)")
 print(total)

 computeVarBTC(total)
}





