<?php 
/*
  This PHP script retrieves a list of job results.
  
  Source code is under GPL, (c) 2002-2016 the Global Processing Unit Team
  
*/
 include('../utils/sql2xml/sql2xml.php');
 include('../utils/sql2xml/xsl.php'); 
 include('../utils/utils.inc.php'); 
 if (getPHPVersion()>=50500) include_once('../utils/mydql2i/mysql2i.class.php');

 
 $jobid = getparam('jobid', "");
 $jobqueueid = getparam('jobqueueid', "");
 $transmissionid = getparam('transmissionid', "");
 if ($jobqueueid!="") {
	$jobclause="and r.jobqueueid='$jobqueueid'"; 
 } else
 if ($jobid!="") {
	$jobclause="and r.jobid='$jobid'"; 
 } else
 if ($transmissionid!="") {
	$jobclause="and q.transmissionid='$transmissionid'";
 } else {
	$jobclause="";
 }
 
 $xml   = getparam('xml', false); 

 if (!$xml) prepare_XSLT();
 echo "<jobresults>\n"; 
 $level_list = Array("jobresult", "jobqueue", "jobdefinition");
 sql2xml("select r.id, r.jobresultid, r.jobdefinitionid, r.jobqueueid, d.job, r.jobresult, r.workunitresult, r.walltime,
                 r.iserroneous, r.errorid, r.errorarg, r.errormsg, 
                 r.nodename, r.nodeid, q.create_dt, q.transmission_dt, q.reception_dt, 
				 d.nodename
         from tbjobresult r, tbjobqueue q, tbjobdefinition d 
         where r.jobqueueid = q.jobqueueid and q.jobdefinitionid = d.jobdefinitionid
		 $jobclause
		 order by r.create_dt desc 
		 LIMIT 0, 40;",
		 $level_list, "13, 3");
 
 echo "</jobresults>\n";
 if (!$xml) apply_XSLT();
 
 ?>