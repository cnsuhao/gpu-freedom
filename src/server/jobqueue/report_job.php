<?php
/*
 This class is used to report a job to the server.
 Internally, the server will create a TBJOBDEFINITION entry and several TBJOBQUEUE entries.
 Later, the server will distribute the TBJOBQUEUE entries to clients willing to compute it.

 Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
*/
include("../utils/utils.inc.php");
include("../utils/constants.inc.php");

$jobid      = getparam('jobid', '');
$jobqueueid = getparam('jobqueueid', '');
$job        = getparam('job', '');
$nodename   = getparam('nodename', '');
$nodeid     = getparam('nodeid', '');
$workunitjob    = getparam('wujob', '');
$workunitresult = getparam('wuresult', '');
$nbrequests     = getparam('nbrequests', '1');
$tagworkunitjob     = getparam('tagwujob', '0');
$tagworkunitresult  = getparam('tagwuresult', '0');
$requireack         = getparam('requireack', '0');
$jobtype            = getparam('jobtype', 'GPU_Engine');

if ( ($jobid=="") || ($job=="") || ($nodename=="") || ($nodeid=="") ) die("ERROR: please specify at least jobid, job, nodename and nodeid");
if ($nbrequests > $max_requests_for_jobs) die ("ERROR: too many requests specified. Maximum is $max_requests_for_jobs");
if ($nbrequests < 1) die("ERROR: at least one request needs to be specified");

$ip = $_SERVER['REMOTE_ADDR'];

include("report_job.inc.php");

$exitmsg = report_job($jobid, $jobqueueid, $job, $nodename, $nodeid, $workunitjob, $workunitresult, $nbrequests, $tagworkunitjob, $tagworkunitresult, $requireack, $jobtype, $ip);

if ($exitmsg == "") echo "OK\n"; else echo "ERROR: $exitmsg\n";
					
?>