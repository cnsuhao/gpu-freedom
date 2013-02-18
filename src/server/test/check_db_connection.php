<?php
/*
  This PHP script reports checks that the connection to the mySQL database
  is up and running
  
  Source code is under GPL, (c) 2002-2011 the Global Processing Unit Team
  
*/

include("../conf/config.inc.php");
mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("ERROR\nUnable to select database, please check settings in conf/config.inc.php\n");

$mainquery  = "SELECT * from tbparameter WHERE paramtype='TEST' and paramname='DB_CONNECTION';"; 
$result=mysql_query($mainquery);
if ($result!="") {
 $num=mysql_numrows($result);
} else $num=0; 

if ($num==0) {
	mysql_close();
	die("ERROR\nTBParameter with paramtype='TEST' and paramname='DB_CONNECTION' does not exist!\n");
}
if ($num>1) {
	mysql_close();
	die("ERROR\nToo many parameters found, is the unique constraint on TBPARAMETER defined?\n");
}
$answer = mysql_result($result,0,"paramvalue");
if ($answer=="OK") {
    echo "OK\n\n";
} else {
    echo "ERROR\n Parameter should be OK but was '$answer'\n";
}

mysql_close();
?>