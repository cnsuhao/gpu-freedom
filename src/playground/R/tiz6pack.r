# tiz6pack, a library with some utils for R
# (c) by 2013 HB9TVM, source code is under GPL

# This function retrieves a measure of smoothness for a given curve
# This is similar concept to Variance.
# Formula is
# smoothness= Sum_i[ (x_i - ((x_i+1+x_i-1)/2)]/n or n-1 if sample is set to true
# test with:
# 
#
#  library(ISwR)
#  snessvar(thuesen$blood.glucose)
#  sness(thuesen$blood.glucose)
snessvar <- function(x, sample=FALSE) {
  N <- length(x)
  xbar <- 1:N
  
  for(i in 1:N) {
       if (i==1) {
			xbar[1]=x[1];
	   } else
	   if (i==N) {
	        xbar[N]=x[N];  
	   } else {
            xbar[i] = (x[i+1] + x[i-1])/2; 
	   }
  }
  
  smoothsst = sum((x - xbar)^2)
  
  if (sample) {
	smoothvar <- smoothsst/(N-1)
  } else {
    smoothvar <- smoothsst/N
  }
  
  return(smoothvar)
}

# Square root of Smoothness
# This is similar to Standard Deviation
sness <- function(x, sample=FALSE) {
   return( sqrt( snessvar(x, sample) ) )
}

# body mass index
bmi <- function(weight, height) {
	return(weight/height^2)
}

analyze <- function(x) {
    cat(" mean:                 ")
	cat(mean(x))
	cat("\n standard deviation: ")
	cat(sd(x))
	cat("\n smoothness:         ")
	cat(sness(x))
	cat("\n")
}