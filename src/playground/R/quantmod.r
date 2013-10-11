# from http://www.quantmod.com/examples/intro/
library(quantmod)
getSymbols("YHOO",src="google") # from google finance
getSymbols("GOOG",src="yahoo") # from yahoo finance 
barChart(GOOG)

getSymbols("XPT/USD",src="oanda") 
chartSeries(XPTUSD,name="Platinum (.oz) in $USD")
# platinum, weekly with candles
chartSeries(to.weekly(XPTUSD),up.col='white',dn.col='blue')

# thechnical analysis
require(TTR)
getSymbols("AAPL")
chartSeries(AAPL)
addMACD()
addBBands()

