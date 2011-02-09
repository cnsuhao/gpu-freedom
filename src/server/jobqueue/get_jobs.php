<?php
/*
  This PHP script stores a job from a client into TBJOB and TBJOBQUEUE
  
  Source code is under GPL, (c) 2002-2011 the Global Processing Unit Team
  
*/
include("../conf/config.inc.php");
$nodeid   = $_GET['nodeid'];
$max      = $_GET['max'];

if (($nodeid=="") || ($max=="")) die('<b>Parameters not defined</b>');
if (($max<1) || ($max>100)) die('<b>Max is too high</b>');

mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("<b>Error: Unable to select database, please check settings in conf/config.inc.php</b>");

$mainquery  = "SELECT * from tbjobqueue WHERE transmitted=0 AND nodeid<>'$nodeid' LIMIT 0,$max;"; 
$result=mysql_query($mainquery);
if ($result!="") {
 $num=mysql_numrows($result);
} else $num=0; 


echo "<jobs>\n";
$i=0;
while ($i<$num) {
   $requestid = mysql_result($result,$i,"id");
   $job_id = mysql_result($result,$i,"job_id");
  
   $jobquery  = "SELECT * from tbjob WHERE id=$job_id"; 
   $jobresult = mysql_query($jobquery);
   
   $jobid     = mysql_result($jobresult,0,"jobid");
   $job       = mysql_result($jobresult,0,"job");
   $workunitincoming = mysql_result($jobresult,0,"workunitincoming");
   $workunitoutgoing = mysql_result($jobresult,0,"workunitoutgoing");
   $requests   = mysql_result($jobresult,0,"requests");
   $delivered  = mysql_result($jobresult,0,"delivered");
   $results    = mysql_result($jobresult,0,"requests");
   $delivered++;
  
   echo "   <job>\n";
   echo "      <externalid>$job_id</externalid>\n";
   echo "      <requestid>$requestid</requestid>\n";
   echo "      <jobid>$jobid</jobid>\n";
   echo "      <job><![CDATA[$job]]></job>";
   echo "      <workunitincoming>$workunitincoming</workunitincoming>\n";
   echo "      <workunitoutgoing>$workunitoutgoing</workunitoutgoing>\n";
   echo "      <requests>$requests</requests>\n";
   echo "      <delivered>$delivered</delivered>\n";
   echo "      <results>$results</results>\n";
   echo "   </job>\n";
   
   $upjobqueuequery  = "UPDATE tbjobqueue SET transmitted=1, transmission_dt=NOW() WHERE id=$requestid"; 
   $upjobqueueresult = mysql_query($upjobqueuequery);
   
   $upjobquery  = "UPDATE tbjob SET delivered=$delivered WHERE id=$job_id"; 
   $upjobresult = mysql_query($upjobquery);
  
  $i++;
}

echo "</jobs>\n";

mysql_close();

?>