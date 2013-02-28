<?php
/*
 This class is used to acknowledge a jobqueue entry to the server.

 Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
*/
include("../utils/utils.inc.php");
include("../utils/constants.inc.php");

$jobqueueid     = getparam('jobqueueid', '');
$jobid          = getparam('jobid', '');
$nodeid         = getparam('nodeid', '');
$nodename       = getparam('nodename', '');

if ( ($jobqueueid=="") || ($jobid=="") || ($nodeid=="") || ($nodename=="") ) die("ERROR: please specify at least jobqueueid, jobid, nodeid and nodename");

$ip = $_SERVER['REMOTE_ADDR'];

include("ack_job.inc.php");

$exitmsg = ack_job($jobqueueid, $jobid, $nodeid, $nodename, $ip);

if ($exitmsg == "") echo "OK"; else echo "ERROR: $exitmsg";
					
?>