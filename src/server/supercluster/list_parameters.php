<?php
/*
  This PHP script retrieves parameters from table TBPARAMETER.
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/

 include('../utils/sql2xml/sql2xml.php'); 
 include('../utils/sql2xml/xsl.php');
 include('../utils/utils.inc.php');
 include('../utils/constants.inc.php');
 
 $paramtype = getparam("paramtype", "");
 if ($paramtype=="") $paramclause = ""; else $paramclause = "and paramtype='$paramtype'";

 $xml = getparam('xml', false); 
 if (!$xml) ob_start();
 
 echo "<parameters>\n"; 
 $level_list = Array("parameter");
 sql2xml("select id, paramtype, paramname, paramvalue, create_dt, update_dt
         from tbparameter
		 where 
		 paramtype<>'CONFIGURATION' and paramtype<>'SECURITY'
		 $paramclause
		 order by paramtype, paramname;
		", $level_list, 0);
 echo "</parameters>\n";
 
 if (!$xml) apply_XSLT();
?>