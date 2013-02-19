<?php
/*
  This PHP script stores a message from a client into TBCHANNEL
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/
include("../conf/config.inc.php");
if (isset($_GET['nodename'])) $nodename = $_GET['nodename']; else $nodename="";
if (isset($_GET['nodeid'])) $nodeid   = $_GET['nodeid'];     else $nodeid="";
if (isset($_GET['user'])) $user     = $_GET['user'];         else $user="";
if (isset($_GET['chantype'])) $chantype = $_GET['chantype']; else $chantype="";
if (isset($_GET['channame'])) $channame = $_GET['channame']; else $channame="";
if (isset($_GET['usertime'])) $usertime = $_GET['usertime']; else $usertime="";
if (isset($_GET['content'])) $content  = $_GET['content'];   else $content="";

if (($nodename=="") || ($nodeid=="") || ($user=="") || ($chantype=="") || ($channame=="") || ($content=="")) die('ERROR: Parameters not defined');

$ip  = $_SERVER['REMOTE_ADDR'];

mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");

$mainquery  = "INSERT INTO tbchannel (id, nodename, nodeid, user, chantype, channame, content, ip, create_dt) 
                               VALUES('', '$nodename','$nodeid','$user','$chantype','$channame','$content', '$ip', NOW());"; 
$result=mysql_query($mainquery);
mysql_close();

echo "OK"
?>