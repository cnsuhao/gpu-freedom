#!/usr/bin/python
import os
import MySQLdb
import SimpleConfigParser

def getCPUtemperature():
	res=os.popen("/opt/vc/bin/vcgencmd measure_temp").readline()
	strtemp=(res.replace("temp=","").replace("'C\n",""))
	#print strtemp
        return strtemp

temp=getCPUtemperature()
print temp

try:
	
        cp = SimpleConfigParser.SimpleConfigParser()
        cp.read('/var/www/html/gpu_freedom/src/playground/bots/raspi/config.ini')

	db=MySQLdb.connect("localhost", cp.getoption("username"), cp.getoption("password"), 
                           cp.getoption("database"))
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
