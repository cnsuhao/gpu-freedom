<?php
/*
 This class is used to report a job to the server.
 Internally, the server will create a TBJOBDEFINITION entry and several TBJOBQUEUE entries.
 Later, the server will distribute the TBJOBQUEUE entries to clients willing to compute it.

*/
include("../utils/utils.inc.php");
include("../utils/constants.inc.php");

$jobid     = getparam('jobid', '');
$job       = getparam('job', '');
$nodename  = getparam('nodename', '');
$nodeid    = getparam('nodeid', '');
$workunitjob    = getparam('wujob', '');
$workunitresult = getparam('wuresult', '');
$nbrequests     = getparam('nbrequests', '1');
$tagworkunitjob     = getparam('tagwujob', '0');
$tagworkunitresult  = getparam('tagwuresult', '0');

if ( ($jobid=="") || ($job=="") || ($nodename=="") || ($nodeid=="") ) die("ERROR: please specify at least jobid, job, nodename and nodeid");
if ($nbrequests > $max_requests_for_jobs) die ("ERROR: too many requests specified. Maximum is $max_requests_for_jobs");

$ip = $_SERVER['REMOTE_ADDR'];

include("report_job.inc.php");

$exitcode = report_job($jobid, $job, $nodename, $nodeid, $workunitjob, $workunitresult, $nbrequests, $tagworkunitjob, $tagworkunitresult, $ip);

if ($exitcode == 1) echo "OK"; else echo "NOT OK";
					
?>