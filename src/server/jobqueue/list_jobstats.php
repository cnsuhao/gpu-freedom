<?php
/*
  This PHP script retrieves the current status of a single job definition,
  it retrieves how many jobqueues there are in a particular status.
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/

 include("../utils/utils.inc.php");	
 include('../utils/sql2xml/sql2xml.php');
 include('../utils/sql2xml/xsl.php'); 
 include("../conf/config.inc.php");	
 if (getPHPVersion()>=50500) include_once('../utils/mydql2i/mysql2i.class.php');

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
	$selectclause = "(d.create_dt > NOW() - INTERVAL 30 DAY)";
 }
 

 if (!$xml) prepare_XSLT();
 $debug=1;	
 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");
 echo "<jobstats>\n";
 
 $queryjobs = "select d.jobdefinitionid, d.job, d.jobtype, d.requireack, d.create_dt
               from tbjobdefinition d
			   where $selectclause;
			  ";
 $resultjobs = mysql_query($queryjobs);
 if ($resultjobs=="") $num=0; else $num=mysql_num_rows($resultjobs); 

 for ($i=0; $i<$num; $i++) {
	$jobdefinitionid  = mysql_result($resultjobs, $i, 'jobdefinitionid');
	$job          = mysql_result($resultjobs, $i, 'job');
	$jobtype      = mysql_result($resultjobs, $i, 'jobtype');
	$requireack   = mysql_result($resultjobs, $i, 'requireack');
	$create_dt    = mysql_result($resultjobs, $i, 'create_dt');
	$nbqueues     = retrieve_count("select count(*) from tbjobqueue where jobdefinitionid='$jobid'");
	$transmitted  = retrieve_count("select count(*) from tbjobqueue where jobdefinitionid='$jobid' and (transmission_dt is not NULL)");
    $acknowledged = retrieve_count("select count(*) from tbjobqueue where jobdefinitionid='$jobid' and (ack_dt is not NULL)");
    $received     = retrieve_count("select count(*) from tbjobqueue where jobdefinitionid='$jobid' and (reception_dt is not NULL)");
	
	echo "<jobstat>\n";
	echo "<jobdefinitionid>$jobdefinitionid</jobdefinitionid>\n";
	echo "<job>$job</job>\n";
	echo "<jobtype>$jobtype</jobtype>\n";
	echo "<requireack>$requireack</requireack>\n";
	echo "<create_dt>$create_dt</create_dt>\n";
	echo "<requests>$nbqueues</requests>\n";
	echo "<transmitted>$transmitted</transmitted>\n";
	echo "<acknowledged>$acknowledged</acknowledged>\n";
	echo "<received>$received</received>\n";
	echo "</jobstat>\n";
	
 }
  
 echo "</jobstats>\n";

 mysql_close();
 if (!$xml) apply_XSLT();
 
?>