<?php
	


 include("../conf/config.inc.php");	
 // defines parameters in the database
 include("../utils/parameters.inc.php");
 
  mysql_connect($dbserver, $username, $password);
 @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php'\n");
 
 // define an unique server id for this installation
 if (get_db_parameter("CONFIGURATION", "SERVER_ID", "missing")=="missing") {
	$c        = uniqid (rand(), true);
    $serverid = md5($c);
    set_db_parameter("CONFIGURATION", "SERVER_ID", $serverid);
 }
 
 if (get_db_parameter("TIME", "UPTIME", "missing")=="missing") {
	set_db_parameter("TIME", "UPTIME", "0");
 }

 if (get_db_parameter("TIME", "TOTAL_UPTIME", "missing")=="missing") {
	set_db_parameter("TIME",  "TOTAL_UPTIME", "0");
 }
 
 if (get_db_parameter("SECURITY", "PWD_HASH_SALT", "missing")=="missing") {
	// TODO: will this be just the next random number after serverid?
	$d        = uniqid (rand(), true);
    $salt     = md5($d);
	set_db_parameter("SECURITY", "PWD_HASH_SALT", $salt);
 }
  
 mysql_close();
 
 echo "OK";
?>