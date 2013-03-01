<?php 
/*
  This PHP script retrieves the current content of the job queue.
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/
 include('../utils/sql2xml/sql2xml.php');
 include('../utils/sql2xml/xsl.php'); 
 include('../utils/utils.inc.php'); 
 include('../utils/constants.inc.php'); 
 include('../conf/config.inc.php'); 
 
 $xml     = getparam('xml', false);
 
 // if the crunch parameter is set, the viewer would like a list of jobs to be computed
 $crunch = getparam('crunch', false); 
 if ($crunch) { 
     // we define an unique id to identify which trades we will transmit
     $transmissionid = create_unique_id();
	 
	 mysql_connect($dbserver, $username, $password);
    @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");
	
	 $query = "update tbjobqueue set transmissionid='$transmissionid', transmission_dt=NOW()
	           where (transmission_dt is NULL) 
			   OR ( (requireack=1) AND (NOW() > transmission_dt + INTERVAL $retransmission_interval SECOND  ) )
			   LIMIT $jobs_to_be_distributed_at_once";
     mysql_query($query);
	 mysql_close();
	 
	 $crunchclause="and transmissionid='$transmissionid'";
	 $limit = $jobs_to_be_distributed_at_once;
 }
 else {
     $crunchclause="";
	 $limit = 40;
 }
 
 if (!$xml) prepare_XSLT();
 echo "<jobqueues>\n"; 
 $level_list = Array("jobqueue", "jobdefinition");
 sql2xml("select q.id, q.jobdefinitionid, q.jobqueueid, q.workunitjob, q.workunitresult, q.nodeid, q.requireack, q.acknodeid, q.acknodename,
                 q.create_dt, q.transmission_dt, q.transmissionid, q.ack_dt, q.reception_dt, d.job,
				 d.nodename
         from tbjobqueue q, tbjobdefinition d 
         where q.jobdefinitionid = d.jobdefinitionid
		 $crunchclause
		 order by q.create_dt desc 
		 LIMIT 0, $limit;"
		 , $level_list, 14);
 
 echo "</jobqueues>\n";
 if (!$xml) apply_XSLT();
 
 ?>