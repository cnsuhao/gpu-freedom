# the goal here is to show that bitcoin prices are lognormal distributed
# using plots and a chi squared test
# (c) 2013 by HB9TVM
o<-order(b_avg_price$usd)
b_avg_price$usd[o]
#plot(b_avg_price$usd[o])
#points(b_avg_price$usd[o]help)
hist(b_avg_price$usd, breaks=12)

N<-length(b_avg_price$usd)
bitcoinm  <- mean(b_avg_price$usd)
bitcoinsd <- sd(b_avg_price$usd)

# there seems to be an error with these two parameters and rlnorm
cmplognormal <- rlnorm(N, meanlog=bitcoinm, sdlog=bitcoinsd)
# rlnorm seems broken to me!
cmpnormal<- rnorm(N, mean=bitcoinm, sd=bitcoinsd)
plot(cmpnormal)
hist(cmpnormal, breaks=12)
print(chisq.test(b_avg_price$usd, cmpnormal))
