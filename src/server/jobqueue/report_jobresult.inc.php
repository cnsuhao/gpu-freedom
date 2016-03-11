<?php
/*
 This class contains the logic to insert a job into the database.
 The job has two parts, a job definition, and an entry in tbjobqueue for each time which needs to be executed.

*/

include_once("../utils/utils.inc.php");
include_once("ack_job.inc.php");
if (getPHPVersion()>=50500) include_once('../utils/mydql2i/mysql2i.class.php');


function check_if_job_already_reported($jobqueueid) {
    $querycheck  = "SELECT count(*) FROM tbjobqueue WHERE jobqueueid='$jobqueueid' AND (reception_dt IS NOT NULL);";
	$resultcheck = mysql_query($querycheck);
	$count = mysql_result($resultcheck, 0, "count(*)");
	if ($count>0) return 0; else return 1;
}

function report_jobresult($jobqueueid, $jobid, $nodeid, $nodename, $jobresult, $workunitresult, $iserroneous, $errorid, $errorarg, $errormsg, $walltime, $ip) {
    include("../conf/config.inc.php");	
    $debug=0;	
	mysql_connect($dbserver, $username, $password);
    @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");

	
	$res = verify_jobqueue_if_exists($jobqueueid, $jobid);
	if ($res==0) { mysql_close(); return "There is no jobqueue entry with the jobid and jobqueueid provided"; } 
	
	// verify that someone else did not already report the result
    $res = check_if_job_already_reported($jobqueueid);	
	if ($res==0) { mysql_close(); return "Someone else already provided an answer for this job!"; } 
	
	// 2. Inserting the job result
	$jobresultid  = create_unique_id();
	$queryinsert = "INSERT INTO tbjobresult (id, jobresultid, jobdefinitionid, jobqueueid, jobresult, workunitresult, iserroneous, errorid, errorarg, errormsg, 
	                                         nodename, nodeid, walltime, ip, create_dt)
									  VALUES('', '$jobresultid', '$jobid', '$jobqueueid', '$jobresult', '$workunitresult', $iserroneous,  $errorid, '$errorarg', '$errormsg',
  									         '$nodename', '$nodeid', $walltime, '$ip', NOW());";
	if ($debug==1) echo "$queryinsert";
	mysql_query($queryinsert);	
	
	// 3. setting reception_dt on jobqueue
	$queryupdate = "UPDATE tbjobqueue SET reception_dt=NOW() WHERE jobqueueid='$jobqueueid';";
	if ($debug==1) echo "$queryupdate";
	mysql_query($queryupdate);
	
	mysql_close();
	return "";
}


?>