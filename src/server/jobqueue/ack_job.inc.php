<?php
/*
 This class contains the logic to insert a job into the database.
 The job has two parts, a job definition, and an entry in tbjobqueue for each time which needs to be executed.

*/

function verify_jobqueue_if_exists($jobqueueid, $jobid) {
	// a connection must be established  
	// Checking if jobqueueid and jobid exist
	$querycount  = "select count(*) from tbjobqueue where jobdefinitionid='$jobid' and jobqueueid='$jobqueueid';";
	$resultcount = mysql_query($querycount);
	$countjobid = mysql_result($resultcount, 0, "count(*)");
	if ($countjobid==1) return 1; else return 0;
}

function verify_if_already_acknowledged($jobqueueid, $jobid) {
    $query  = "select ack_dt from tbjobqueue where jobdefinitionid='$jobid' and jobqueueid='$jobqueueid';";
	$result = mysql_query($query);
	$ack_dt = mysql_result($result, 0, "ack_dt");
	if ($ack_dt=="") return 1; else return 0;
	
}

function ack_job($jobqueueid, $jobid, $nodeid, $nodename, $ip) {
    include("../conf/config.inc.php");	
    	
	mysql_connect($dbserver, $username, $password);
    @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");

	$res = verify_jobqueue_if_exists($jobqueueid, $jobid);
	if ($res==0) return "There is no jobqueue entry with the jobid and jobqueueid provided"; 

	$queryupdate  = "update tbjobqueue set acknodeid='$nodeid', acknodename='$nodename', ack_dt=NOW() where jobdefinitionid='$jobid' and jobqueueid='$jobqueueid';";
	mysql_query($queryupdate);
	
	mysql_close();
	return "";
}


?>