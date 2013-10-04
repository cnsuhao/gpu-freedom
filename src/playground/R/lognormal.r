# the goal here is to show that bitcoin prices are lognormal distributed
# using plots and a chi squared test
# (c) 2013 by HB9TVM
o<-order(b_price$price)
b_price$price[o]
#plot(b_avg_price$usd[o])
#points(b_avg_price$usd[o]help)
#hist(b_price$price, breaks=12)

x=b_price$price
N=length(x)
m= mean(x)
v= var(x)

# from http://www.mathworks.ch/ch/help/stats/lognrnd.html
# Generate one million lognormally distributed random numbers with mean m and sigma s
mu = log((m^2)/sqrt(v+m^2));
sigma = sqrt(log(v/(m^2)+1));

# normal and lognormal series
cmplognormal <- rlnorm(N, meanlog=mu, sdlog=sigma)
cmpnormal<- rnorm(N, mean=m, sd=sd(x))

# plots
hist(cmpnormal, breaks=12, col = "lightyellow", border = "pink", main="", xlab="", ylab="")
hist(cmplognormal, breaks=12, col = "lightblue", border = "pink", add=T, main="", xlab="", ylab="")
hist(x,col= "gray", add=T, main="", xlab="", ylab="")
title(main="Bitcoin Price Distribution", xlab="USD", ylab="freq")

# chi squared test
print(chisq.test(x, cmplognormal))

