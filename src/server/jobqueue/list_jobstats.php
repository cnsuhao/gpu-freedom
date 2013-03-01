<?php
 include("../utils/utils.inc.php");	
 include('../utils/sql2xml/sql2xml.php');
 include('../utils/sql2xml/xsl.php'); 
 include("../conf/config.inc.php");	

 function retrieve_count($query) {
	// a connection needs to be established
	$result = mysql_query($query);
	$count = mysql_result($result, 0, "count(*)");
	return $count;
 } 
 
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
 
 $queryjobs = "select d.jobdefinitionid, d.job, d.requireack
               from tbjobdefinition d
			   where $selectclause;
			  ";
 $resultjobs = mysql_query($queryjobs);
 if ($resultjobs=="") $num=0; else $num=mysql_numrows($resultjobs); 

 for ($i=0; $i<$num; $i++) {
	$jobid        = mysql_result($resultjobs, $i, 'jobdefinitionid');
	$job          = mysql_result($resultjobs, $i, 'job');
	$requireack   = mysql_result($resultjobs, $i, 'requireack');
	$nbqueues = retrieve_count("select count(*) from tbjobqueue where jobdefinitionid='$jobid'");
    
	echo "<job>\n";
	echo "<jobid>$jobid</jobid>\n";
	echo "<job>$job</job>\n";
	echo "<requireack>$requireack</requireack>\n";
	echo "<queues>$nbqueues</queues>\n";
	echo "</job>\n";
	
 }
  
 echo "</jobstats>\n";

 mysql_close();
 if (!$xml) apply_XSLT();
 
?>