<?php

function report_job($jobid, $job, $nodename, $nodeid, $workunitjob, $workunitresult, $nbrequests, $tagworkunitjob, $tagworkunitresult, $ip) {
    include("../conf/config.inc.php");	
    $debug=0;	
	mysql_connect($dbserver, $username, $password);
    @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");


	mysql_close();
}


?>