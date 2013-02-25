<?php
/*
  This PHP script reports checks that the connection to the mySQL database
  is up and running
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/

include("../utils/constants.inc.php");

if (!file_exists("../conf/config.inc.php")) {
    die("ERROR: Configuration file conf/config.inc.php is missing. Please setup GPU server'\n");
 }  
 
include("../conf/config.inc.php");
include("../utils/parameters.inc.php");


mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php'\n");

$answer = get_db_parameter("TEST", "DB_CONNECTION", "NOT OK");

mysql_close();

echo "$answer";
?>