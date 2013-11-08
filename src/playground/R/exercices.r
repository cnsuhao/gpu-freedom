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
qqnorm(b_avg_price$usd)

#boxplot
par(mfrow=c(1,2))
boxplot(IgM)
boxplot(log(IgM))
par(mfrow=c(1,1))

#t-test, testing if a sample has a particular mean mu
daily.intake <- c(5260, 5470, 5640, 6180, 6390, 6515, 6805, 7515, 7515, 8230, 8770)
mean(daily.intake)
sd(daily.intake)
quantile(daily.intake)
t.test(daily.intake, mu=7725)

#same as t.test, but no assumption on normal distribution, distribution simply symmetric around mu_0
#expend is described by stature: expend~stature 
attach(energy)
t.test(expend~stature)
# the test is significant p<0.05, therefore expend is not described by stature

#testing the variances
var.test(expend~stature)
# there is no evidence against the assumption that the two distributions have the same variance

attach(intake)
intake
post-pre
#paired=T if both measurements types were taken at the same time for the same patient
t.test(pre, post, paired=T)
wilcox.test(pre,post, paired=T)

#linear regression
attach(thuesen)
object.model<-lm(short.velocity~blood.glucose)
#short.velocity=1.098+0.022*blood.glucose
summary(object.model)
plot(blood.glucose,short.velocity)
abline(object.model)
fitted(object.model)
resid(object.model)

#correlation
cor(blood.glucose,short.velocity)
cor(blood.glucose,short.velocity,use="complete.obs")
cor(thuesen,use="complete.obs")
cor.test(blood.glucose, short.velocity)

#advanced data handling
age <- subset(juul, age>=10 & age <=16)$age
range(age)
length(age)
agegroup <- cut(age, seq(10,16,2), right=F, include.lowest=T)
table(agegroup)

q<-quantile(age, c(0, 0.25, 0.5, 0.75, 1))
ageQ <- cut(age, q, include.lowest=T)
table(ageQ)

levels(ageQ) <- c("1st", "2nd", "3rd", "4th")
levels(agegroup) <- c("10-11","12-13","14-15")
table(ageQ)
table(agegroup)

#working with dates
stroke <- read.csv2( system.file("rawdata","stroke.csv", package="ISwR"), na.strings=".")
names(stroke) <- tolower(names(stroke))
head(stroke)

stroke <- transform(stroke, died=as.Date(died, format("%d.%m.%Y")), dstr=as.Date(died, format("%d.%m.%Y")))
summary(stroke$died)
summary(stroke$dstr)



