<?php
/*
  This PHP script stores a job from a client into TBJOB and TBJOBQUEUE
  
  Source code is under GPL, (c) 2002-2011 the Global Processing Unit Team
  
*/
include("../conf/config.inc.php");
$nodeid   = $_GET['nodeid'];
$nodename = $_GET['nodename'];
$jobid    = $_GET['jobid'];
$job      = $_GET['job'];
$workunitincoming = $_GET['workunitincoming'];
$workunitoutgoing = $_GET['workunitoutgoing'];
$requests = $_GET['requests'];
$ip       = $_SERVER['REMOTE_ADDR'];

if (($nodeid=="") || ($jobid=="") || ($job=="") || ($workunitincoming=="") || ($workunitoutgoing=="") || ($requests=="")) die('<b>Parameters not defined</b>');
if (($requests<1) || ($requests>1000)) die('<b>Too many requests</b>');

mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("<b>Error: Unable to select database, please check settings in conf/config.inc.php</b>");

$mainquery  = "INSERT INTO tbjob (id, jobid, job, workunitincoming, workunitoutgoing, requests, nodename, nodeid, ip, create_dt) 
               VALUES('', '$jobid', '$job', '$workunitincoming', '$workunitoutgoing', $requests, '$nodename', '$nodeid', '$ip', NOW());"; 
$result=mysql_query($mainquery);

// retrievieng the jobid
$selquery  = "SELECT id FROM tbjob WHERE (jobid='$jobid') AND (nodeid='$nodeid') AND (job='$job');"; 
$selresult = mysql_query($selquery);
if ($selresult=="") {
   die('<b>Internal error: jobid $jobid does not exist on server!</b>');
   mysql_close();
}
$job_id    = mysql_result($selresult,0,"id");

$i=0;
while ($i<$requests) {

    $jqquery  = "INSERT INTO tbjobqueue (id, job_id, nodeid, create_dt) 
                 VALUES('', $job_id, '$nodeid', NOW());"; 
    $jqresult=mysql_query($jqquery);

	$i++;
}

echo "<report>\n";
echo "   <job>\n";
echo "      <externalid>$job_id</externalid>\n";
echo "   </job>\n";
echo "</report>\n";

mysql_close();

?>