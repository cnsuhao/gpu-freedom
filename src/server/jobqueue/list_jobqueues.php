<?php 
/*
  This PHP script retrieves the current content of the job queue.
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/
 include('../utils/sql2xml/sql2xml.php');
 include('../utils/sql2xml/xsl.php'); 
 include('../utils/utils.inc.php'); 
 
 $xml = getparam('xml', false); 

 if (!$xml) prepare_XSLT();
 echo "<jobqueues>\n"; 
 $level_list = Array("jobqueue", "jobdefinition");
 sql2xml("select q.id, q.jobdefinitionid, q.jobqueueid, q.workunitjob, q.workunitresult, q.nodeid,
                 q.create_dt, q.transmission_dt, q.reception_dt, d.job,
				 d.nodename
         from tbjobqueue q, tbjobdefinition d 
         where q.jobdefinitionid = d.jobdefinitionid
		 order by q.create_dt desc 
		 LIMIT 0, 40;"
		 , $level_list, 9);
 
 echo "</jobqueues>\n";
 if (!$xml) apply_XSLT("jobqueue");
 
 ?>