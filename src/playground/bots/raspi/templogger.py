#!/usr/bin/python
import os
import MySQLdb
def getCPUtemperature():
	res=os.popen("/opt/vc/bin/vcgencmd measure_temp").readline()
	strtemp=(res.replace("temp=","").replace("'C\n",""))
	#print strtemp
        return strtemp

temp=getCPUtemperature()
print temp

try:
	db=MySQLdb.connect("localhost", "raspi", "pwd$2016", "raspi")
	c=db.cursor()
        sqlstr="INSERT INTO tbtemperature (insert_dt, temperature_raspi) VALUES(NOW(), "+temp+");"
        print sqlstr
        c.execute(sqlstr)        
        db.commit()
	db.close()
except:
	print "ERROR: connecting to database"
	db.close()
	exit(1)
