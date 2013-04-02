<?php
/*
  This PHP script retrieves the current status of a single job queue.
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/
 include("../conf/config.inc.php");	

 $jobqueueid = getparam('jobqueueid', "");
 if ($jobqueueid=="") die("ERROR: parameter jobqueueid is not defined");

 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");
 
 $query = "select q.jobdefinitionid, q.jobqueueid, q.acknodename,
                  q.create_dt, q.transmission_dt, q.transmissionid, q.ack_dt, q.reception_dt, d.nodename
                  from tbjobqueue q, tbjobdefinition d 
                  where q.jobdefinitionid = d.jobdefinitionid and q.jobqueueid='$jobqueueid';";
 $result = mysql_query($query);
 
 $timestamp=time();
 if ($result=="") {
	$status="ERROR";
	$nodename="";
	$message="Jobqueueid $jobqueueid does not exist on $my_server_url";
 } else {
		// retrieve everything from jobqueue;
		$acknodename     = mysql_result($result, 0, 'acknodename');
		$create_dt       = mysql_result($result, 0, 'create_dt');
		$transmission_dt = mysql_result($result, 0, 'transmission_dt');
		$transmissionid  = mysql_result($result, 0, 'tranmissionid');
		$ack_dt          = mysql_result($result, 0, 'ack_dt');
		$reception_dt    = mysql_result($result, 0, 'reception_dt');
		$nodename        = mysql_result($result, 0, 'nodename');
		
		if ($reception_dt!="") {
			// we received a result for this job
			// retrieving the corresponding jobresult
			$queryjobresult = "select r.jobresultid, r.nodename from tbjobresult r where r.jobqueueid='$jobqueueid'";
			$resjobresult   = mysql_query($queryjobresult); 
			if ($resjobresult=="") {
				$status="ERROR";
				$nodename="";
				$message="Internal server error: For jobqueueid $jobqueueid there is no jobresult defined, although reception_dt is set";
			} else {
				$nodename = mysql_result($result, 0, 'nodename');
			    $message  = mysql_result($result, 0, 'jobresultid');
				$message  = "jobresultid=" . $message;
				$timestamp = $reception_dt;
				$status    = "COMPLETED"
			}	
		} else
		if ($ack_dt!="") {
			$nodename=$acknodename;
			$message="";
			$timestamp=$ack_dt;
			$status="ACKNOWLEDGED";
		} else
		if ($transmission_dt!="") {
			$nodename="";
			$message="Transmitted with transmissionid=$transmissionid";
			$timestamp=$transmitted_dt;
			$status="TRANSMITTED";
		}
		else 
		if (create_dt!="") {
			$nodename="";
			$message="";
			$timestamp=$create_dt;
			$status="NEW";
		} else {
			$nodename="";
			$message="Internal error in status_jobqueue.php: create_dt not set";
			$status="ERROR";
		}
 
 }
  
 // output part
 echo "<jobstatus>\n";
 echo "<status>$status</status>\n";
 echo "<timestamp>$timestamp</timestamp>\n";
 echo "<nodename>$nodename</nodename>\n";
 echo "<message>$message</message>\n";
 echo "</jobstatus>\n";
?>