<?php
/*
  This PHP script stores a result for a job from a client into TBJOBRESULT
  
  Source code is under GPL, (c) 2002-2011 the Global Processing Unit Team
  
*/

include("../conf/config.inc.php");
$nodeid    = $_GET['nodeid'];
$nodename  = $_GET['nodename'];
$jobid     = $_GET['jobid'];
$requestid = $_GET['requestid'];
$jobresult = $_GET['jobresult'];
$workunitresult = $_GET['workunitresult'];
$iserroneous    = $_GET['iserroneous'];
$errorid   = $_GET['errorid'];
$errormsg  = $_GET['errormsg'];
$errorarg  = $_GET['errorarg'];
$ip        = $_SERVER['REMOTE_ADDR'];

if (($nodeid=="") || ($jobid=="") || ($requestid=="") || ($iserroneous=="") || ($errorid=="")) die('<b>Parameters not defined</b>');

mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("<b>Error: Unable to select database, please check settings in conf/config.inc.php</b>");

// retrieving the job id
$selquery  = "SELECT * FROM tbjob WHERE (jobid='$jobid');"; 
$selresult = mysql_query($selquery);
if ($selresult=="") {
   die('<b>Jobid $jobid does not exist in table TBJOB on server!</b>');
   mysql_close();
}
$job_id    = mysql_result($selresult,0,"id");
$results   = mysql_result($selresult,0,"results"); 

// checking that requestid exists
$selquery  = "SELECT id FROM tbjobqueue WHERE (id='$requestid');"; 
$selresult = mysql_query($selquery);
if ($selresult=="") {
   die('<b>Requestid $requestid does not exist as primary key in table TBJOBQUEUE on server!</b>');
   mysql_close();
}

// inserting mainresult
$mainquery  = "INSERT INTO tbjobresult (id, jobid, jobresult, workunitresult, iserroneous, errorid, errormsg, errorarg, jobqueue_id, nodename, nodeid, ip, create_dt) 
               VALUES('', '$jobid', '$jobresult', '$workunitresult', $iserroneous, $errorid, '$errormsg', '$errorarg', $requestid, '$nodename', '$nodeid', '$ip', NOW());"; 
$result=mysql_query($mainquery);

// updating jobqueue
$upjobqueuequery  = "UPDATE tbjobqueue SET received=1, reception_dt=NOW() WHERE id=$requestid"; 
$upjobqueueresult = mysql_query($upjobqueuequery);
 
// updating job 
$results++;
$upjobquery  = "UPDATE tbjob SET results=$results WHERE id=$job_id"; 
$upjobresult = mysql_query($upjobquery);

mysql_close();

?>