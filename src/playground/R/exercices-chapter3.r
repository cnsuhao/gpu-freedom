
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
plot(x, dnorm(x), type="l")
curve(dnorm(x), from=-4, to=4)
curve(dnorm(x), from=-4, to=4)
x<-0:50
curve(pnorm(x), from=-4, to=4)
curve(qnorm(x), from=-4, to=4)
curve(rnorm(x), from=-4, to=4)
plot(x,dbinom(x,size=50,prob=.33),type="h")
