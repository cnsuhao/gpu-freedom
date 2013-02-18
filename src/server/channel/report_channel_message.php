<?php
/*
  This PHP script stores a message from a client into TBCHANNEL
  
  Source code is under GPL, (c) 2002-2011 the Global Processing Unit Team
  
*/
include("../conf/config.inc.php");
$nodename = $_GET['nodename'];
$nodeid   = $_GET['nodeid'];
$user     = $_GET['user'];
$chantype = $_GET['chantype'];
$channame = $_GET['channame'];
$usertime = $_GET['usertime'];
$content  = $_GET['content'];
$ip       = $_SERVER['REMOTE_ADDR'];

if (($nodename=="") || ($nodeid=="") || ($user=="") || ($chantype=="") || ($channame=="") || ($content=="")) die('<b>Parameters not defined</b>');

mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("<b>Error: Unable to select database, please check settings in conf/config.inc.php</b>");

$mainquery  = "INSERT INTO tbchannel (id, nodename, nodeid, user, chantype, channame, content, ip, create_dt) 
                               VALUES('', '$nodename','$nodeid','$user','$chantype','$channame','$content', '$ip', NOW());"; 
$result=mysql_query($mainquery);
mysql_close();

?>