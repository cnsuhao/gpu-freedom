# the goal here is to show that bitcoin prices are lognormal distributed
# using plots and a chi squared test
# (c) 2013 by HB9TVM
o<-order(b_price$price)
b_price$price[o]
#plot(b_avg_price$usd[o])
#points(b_avg_price$usd[o]help)
hist(b_price$price, breaks=12)

x <- b_price$price
N<-length(x)
bitcoinm  <- mean(x)
bitcoinsd <- sd(x)

# from https://stat.ethz.ch/pipermail/r-help/2008-May/161229.html
# A mean of 100 for the log-normal variate?  In this case any set of mu
# and sd for which exp(mu+sd^2/2)=100 (or mu+sd^2/2=log(100)) would do
# the trick
logmu = log(bitcoinm)/2
logsd = sqrt(log(bitcoinm))

cmplognormal <- rlnorm(N, meanlog=logmu, sdlog=logsd)
# rlnorm seems broken to me!
cmpnormal<- rnorm(N, mean=bitcoinm, sd=bitcoinsd)
plot(cmpnormal)
hist(cmpnormal, breaks=12)
print(chisq.test(x, cmpnormal))
