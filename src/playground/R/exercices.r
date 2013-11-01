library(ISwR)
print
methods(print)
aq <- edit(airquality)
help(airquality)
aq <- edit(airquality)
sample(1:40, 5)
sample(1:10,4,replace=T)
sample(c("succ","fail"), 10, replace="T", prob=c(0.9,0.1))
sample(c("succ","fail"), 10, replace=T, prob=c(0.9,0.1))
1/prod(40:35)
1/prod(40:36)
5!
prod(5:1)/prod(40:36)
choose(40,5)
1/choose(40,5)
x<-seq(-4,4,.1)
x

# density function dnorm()
# cumulative function pnorm()
# quantile (inverse of cumulative) qnorm()
# pseudo random numbers with rnorm

# some with ?binom, e.g. rbinom for pseudo random numbers
plot(x, dnorm(x), type="l")
curve(dnorm(x), from=-4, to=4)
curve(dnorm(x), from=-4, to=4)
x<-0:50
curve(pnorm(x), from=-4, to=4)
curve(qnorm(x), from=-4, to=4)
curve(rnorm(x), from=-4, to=4)
plot(x,dbinom(x,size=50,prob=.33),type="h")


x<-rnorm(50)
mean(x)
sd(x)
var(x)
median(x)
quantile(x)

pvec <- seq(0,1,0.1)
quantile(x, pvec)
summary(x)

# accidents per age
mid.age <- c(2.5,7.5,13,16.5,17.5,19,22.5,44.5,70.5)
acc.count <- c(28,46,58,20,31,64,149,316,103)
age.acc <- rep(mid.age, acc.count)

brk <- c(0,5,10,16,17,18,20,25,60,80)
hist(age.acc,breaks=brk)
hist(age.acc,breaks=brk, freq=T)

#empirical cumulative distribution
n<-length(x)
plot(sort(x), (1:n)/n,type="s",ylim=c(0,1))

#Q-Q plots
#you get a straight line if data is normally distributed
qqnorm(x)
# bitcoin data after sourcing from jdbc.r
qqnorm(b_price$price)

#boxplot
par(mfrow=c(1,2))
boxplot(IgM)
boxplot(log(IgM))
par(mfrow=c(1,1))





