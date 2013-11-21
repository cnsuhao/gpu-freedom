library(PerformanceAnalytics)

percentdiff <- function(x) {
  N <- length(x)
  xdiff <- numeric(N-1)
  
  for(i in 1:N-1) {
 			xdiff[i] = (x[i+1]/x[i]) - 1;
  }
  return(xdiff)
}


computeVar <- function(btc_open, btc_price) {
  #requires jdbc.r to be loaded
  pct <- percentdiff(b_avg_price_last_60$usd)
  return(VaR(pct, p = 0.95)[1]*btc_open*btc_price)
}  


# source("C:\\xampp\\htdocs\\gpu_freedom\\src\\playground\\R\\jdbc.r")
# source("C:\\xampp\\htdocs\\gpu_freedom\\src\\playground\\R\\performanceanalytics.r")
# I have open 0.3 BTC and the current price is 500$
# computeVar(0.3,500)
# it returns the amount of money i can loose on the next day 
# with 95% confidence level and 60 days time horizon in the past



