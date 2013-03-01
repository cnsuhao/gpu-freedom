<?php

 function retrieve_count($table, $idname, $idvalue, $whereclause) {
	// a connection needs to be established

 }
 include("../utils/utils.inc.php");	
 include("../conf/config.inc.php");	
 
 $xml   = getparam('xml', false); 
 $jobid = getparam('jobid', "");
 $transmissionid = getparam('transmissionid', "");
 
 if ($transmissionid!="") {
	$selectclause = "(d.transmissionid='$transmissionid')";
 } else
 if ($jobid != "") {
    $selectclause = "(d.jobdefinitionid='$jobid')";
 } else {
	$selectclause = "(d.create_dt > NOW() - INTERVAL 7 DAY)";
 }
 

 if (!$xml) prepare_XSLT();
 $debug=1;	
 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");
 echo "<jobstats>\n";
 
 $queryjobs = "select d.jobdefinitionid, d.job
               from tbjobdefinition d
			   where $selectclause;
			  ";
 echo "$queryjobs";
 
 
 
 
 echo "</jobstats>\n";

 mysql_close();
 if (!$xml) apply_XSLT();
 
?>