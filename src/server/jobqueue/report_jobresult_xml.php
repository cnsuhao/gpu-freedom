<?php
/*
  This PHP script stores a result for a job from a client into TBJOBRESULT
  
  Source code is under GPL, (c) 2002-2011 the Global Processing Unit Team
  
*/

include("../conf/config.inc.php");
$nodeid    = $_GET['nodeid'];
$nodename  = $_GET['nodename'];
$jobid     = $_GET['jobid'];
$jobresult = $_GET['jobresult'];
$workunitresult = $_GET['workunitresult'];
$iserroneous    = $_GET['iserroneous'];
$errorid   = $_GET['errorid'];
$errormsg  = $_GET['errormsg'];
$errorarg  = $_GET['errorarg'];
$ip        = $_SERVER['REMOTE_ADDR'];

if (($nodeid=="") || ($jobid=="") || ($jobresult=="") || ($workunitresult=="") || ($iserroneous=="") || ($errorid=="")) die('<b>Parameters not defined</b>');

mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("<b>Error: Unable to select database, please check settings in conf/config.inc.php</b>");

// retrievieng the job id
$selquery  = "SELECT id FROM tbjob WHERE (jobid='$jobid');"; 
$selresult = mysql_query($selquery);
$job_id    = mysql_result($selresult,0,"id");

mysql_close();

?>