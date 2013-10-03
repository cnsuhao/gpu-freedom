# the goal here is to show that bitcoin prices are lognormal distributed
# using plots and a chi squared test
# (c) 2013 by HB9TVM
o<-order(b_price$price)
b_price$price[o]
#plot(b_avg_price$usd[o])
#points(b_avg_price$usd[o]help)
hist(b_price$price, breaks=12)

x <- b_price$price/100
N<-length(b_price$price)
bitcoinm  <- mean(x)
bitcoinsd <- sd(x)

# there seems to be an error with these two parameters and rlnorm
cmplognormal <- rlnorm(N, meanlog=exp(bitcoinm), sdlog=exp(bitcoinsd))
# rlnorm seems broken to me!
cmpnormal<- rnorm(N, mean=bitcoinm, sd=bitcoinsd)
plot(cmpnormal)
hist(cmpnormal, breaks=12)
print(chisq.test(x, cmpnormal))
