<?php
/*
 This class contains the logic to insert a job into the database.
 The job has two parts, a job definition, and an entry in tbjobqueue for each time which needs to be executed.

*/

include_once("../utils/utils.inc.php");
include_once("ack_job.inc.php");

function report_jobresult($jobqueueid, $jobid, $nodeid, $nodename, $jobresult, $workunitresult, $iserroneous, $errorid, $errorarg, $errormsg, $ip) {
    include("../conf/config.inc.php");	
    $debug=1;	
	mysql_connect($dbserver, $username, $password);
    @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");

	
	$res = verify_jobqueue_if_exists($jobqueueid, $jobid);
	if ($res==0) { mysql_close(); return "There is no jobqueue entry with the jobid and jobqueueid provided"; } 
	
	$jobresultid  = create_unique_id();
	
	// 2. Inserting the job result
	$queryinsert = "INSERT INTO tbjobresult (id, jobresultid, jobdefinitionid, jobqueueid, jobresult, workunitresult, iserroneous, errorid, errorarg, errormsg, 
	                                         nodename, nodeid, ip, create_dt)
									  VALUES('', '$jobresultid', '$jobid', '$jobqueueid', '$jobresult', '$workunitresult', $iserroneous,  $errorid, '$errorarg', '$errormsg',
  									         '$nodename', '$nodeid', '$ip', NOW());";
	if ($debug==1) echo "$queryinsert";
	mysql_query($queryinsert);	
	
	mysql_close();
	return "";
}


?>