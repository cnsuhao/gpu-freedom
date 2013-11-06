library(moonsun)
jd()
jd(2013,11,5)
gmt()
jday=jd()+gmt()/24
moon(jday)
phase = moon()$phase


