<?php
/*
 This class contains the logic to insert a job into the database.
 The job has two parts, a job definition, and an entry in tbjobqueue for each time which needs to be executed.

*/

include_once("../utils/utils.inc.php");
if (getPHPVersion()>=50500) include_once('../utils/mydql2i/mysql2i.class.php');


function report_job($jobid, $jobqueueid, $job, $nodename, $nodeid, $workunitjob, $workunitresult, $nbrequests, $tagworkunitjob, $tagworkunitresult, $requireack, $jobtype, $ip) {
    include("../conf/config.inc.php");	
    $debug=0;	
	if (($jobqueueid!="") && ($nbrequests>1)) die("ERROR: jobqueueid is provided, but the number of requests is greater than one!");
			
	mysql_connect($dbserver, $username, $password);
    @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");

	// 1. Checking if jobid is unique
	$querycount  = "select count(*) from tbjobdefinition where jobdefinitionid='$jobid';";
	$resultcount = mysql_query($querycount);
	$countjobid = mysql_result($resultcount, 0, "count(*)");
	if ($countjobid>0) { mysql_close(); return "Jobid is not unique! Please choose another jobid!"; }

	// 2. Inserting the job definition
	$queryjobinsert = "INSERT INTO tbjobdefinition (id, jobdefinitionid, job, nodename, nodeid, jobtype, requireack, ip, create_dt, update_dt)
					   VALUES('', '$jobid', '$job', '$nodename', '$nodeid', '$jobtype', $requireack, '$ip', NOW(), NOW());";
	if ($debug==1) echo "$queryjobinsert";
	mysql_query($queryjobinsert);
	
	// 3. Inserting Jobqueue entries, one for each request
	for ($i=0; $i<$nbrequests; $i++) {
			// generate unique jobqueueid, if not provided by client
			if ($jobqueueid=="") $jobqueueid  = create_unique_id();
			
			if ($tagworkunitjob==1)    $wujob="$workunitjob"."_"."$i"; else $wujob="$workunitjob";
			if ($tagworkunitresult==1) $wures="$workunitresult"."_"."$i"; else $wures="$workunitresult";
	
			$queryjobqueue = "INSERT INTO tbjobqueue (id, jobdefinitionid, jobqueueid, workunitjob, workunitresult, nodeid, nodename, requireack, ip, create_dt)
					          VALUES('', '$jobid', '$jobqueueid', '$wujob', '$wures', '$nodeid', '$nodename', $requireack, '$ip', NOW());";
			if ($debug==1) echo "$queryjobqueue";
			mysql_query($queryjobqueue);
	}
	
	mysql_close();
	return "";
}


?>