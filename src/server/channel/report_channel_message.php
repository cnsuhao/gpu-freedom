<?php
/*
  This PHP script stores a message from a client into TBCHANNEL
  
  Source code is under GPL, (c) 2002-2016 the Global Processing Unit Team
  
*/
include("../utils/utils.inc.php");
include("../conf/config.inc.php");
 if (getPHPVersion()>=50500) include_once('../utils/mydql2i/mysql2i.class.php');


$nodename = getparam('nodename', '');
$nodeid   = getparam('nodeid', '');
$user     = getparam('user', '');
$chantype = getparam('chantype', '');
$channame = getparam('channame', '');
$usertime = getparam('usertime', '');
$content  = getparam('content', '');

if (($nodename=="") || ($nodeid=="") || ($user=="") || ($chantype=="") || ($channame=="") || ($content=="")) die('ERROR: Parameters not defined');

$ip  = $_SERVER['REMOTE_ADDR'];

mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");

$mainquery  = "INSERT INTO tbchannel (id, nodename, nodeid, user, chantype, channame, content, ip, create_dt) 
                               VALUES('', '$nodename','$nodeid','$user','$chantype','$channame','$content', '$ip', NOW());"; 
$result=mysql_query($mainquery);
mysql_close();

echo "OK\n"
?>