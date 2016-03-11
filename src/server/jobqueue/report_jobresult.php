<?php
/*
 This class is used to report a job result to the server.

 Source code is under GPL, (c) 2002-2016 the Global Processing Unit Team
*/
include("../utils/utils.inc.php");
include("../utils/constants.inc.php");
if (getPHPVersion()>=50500) include_once('../utils/mydql2i/mysql2i.class.php');


$jobqueueid     = getparam('jobqueueid', '');
$jobid          = getparam('jobid', '');
$nodeid         = getparam('nodeid', '');
$nodename       = getparam('nodename', '');
$jobresult      = getparam('jobresult', '');
$workunitresult = getparam('wuresult', '');
$iserroneous    = getparam('iserroneous', '');
$errorid		= getparam('errorid', 0);
$errorarg		= getparam('errorarg', '');
$errormsg		= getparam('errormsg', '');
$walltime		= getparam('errormsg', 0);

if ( ($jobqueueid=="") || ($jobid=="") || ($nodeid=="") || ($nodename=="") || ($iserroneous=="") ) die("ERROR: please specify at least jobqueueid, jobid, nodeid, nodename and iserroneous");

$ip = $_SERVER['REMOTE_ADDR'];

include("report_jobresult.inc.php");

$exitmsg = report_jobresult($jobqueueid, $jobid, $nodeid, $nodename, $jobresult, $workunitresult, $iserroneous, $errorid, $errorarg, $errormsg, $walltime, $ip);

if ($exitmsg == "") echo "OK\n"; else echo "ERROR: $exitmsg\n";
					
?>