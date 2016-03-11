<?php
/*
  This PHP script retrieves the latest job definitions.
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/

 include('../utils/sql2xml/sql2xml.php'); 
 include('../utils/sql2xml/xsl.php');
 include('../utils/utils.inc.php');
 include('../utils/constants.inc.php');
 if (getPHPVersion()>=50500) include_once('../utils/mydql2i/mysql2i.class.php');
 
 $jobdefinitionid = getparam("jobdefinitionid", "");
 if ($jobdefinitionid=="") $jobdefinitionclause = ""; else $jobdefinitionclause = "and jobdefinitionid='$jobdefinitionid'";

 $xml = getparam('xml', false); 
 if (!$xml) ob_start();
 
 echo "<jobdefinitions>\n"; 
 // TODO: enable condition selection only the job definitions of the last day
 $level_list = Array("jobdefinition");
 sql2xml("select id, jobdefinitionid, job, jobtype, nodename,  create_dt, update_dt
          from tbjobdefinition 
		  where (1=1)
         -- and update_dt >= ( curdate() - interval 1 day ) 
		 $jobdefinitionclause
		 order by update_dt desc 
		 limit 0, 100;
		", $level_list, 0);
 echo "</jobdefinitions>\n";
 
 if (!$xml) apply_XSLT();
?>