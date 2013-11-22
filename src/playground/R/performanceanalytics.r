library(PerformanceAnalytics)

percentdiff <- function(x) {
  N <- length(x)
  xdiff <- numeric(N-1)
  
  for(i in 1:N-1) {
 			xdiff[i] = (x[i+1]/x[i]) - 1;
  }
  return(xdiff)
}


# general compute value at risk
computeVar <- function(x, openpos, price) {
  pct <- percentdiff(x)
  return(VaR(pct, p = 0.95)[1]*openpos*price)
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





