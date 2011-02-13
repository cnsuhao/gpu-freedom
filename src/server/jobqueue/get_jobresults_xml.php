<?php
/*
  This PHP script retrieves jobresults for a particular job id
  
  Source code is under GPL, (c) 2002-2011 the Global Processing Unit Team
  
*/
include("../conf/config.inc.php");
$jobid   = $_GET['jobid'];
if ($jobid=="")  die('<b>Parameters not defined</b>');

mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("<b>Error: Unable to select database, please check settings in conf/config.inc.php</b>");

$mainquery  = "SELECT * from tbjobresult WHERE jobid='$jobid'"; 
$result=mysql_query($mainquery);
if ($result!="") {
 $num=mysql_numrows($result);
} else $num=0; 


echo "<jobresults>\n";
$i=0;
while ($i<$num) {
   $externalid     = mysql_result($result,$i,"id");
   $jobid          = mysql_result($result,0,"jobid");
   $jobresult      = mysql_result($result,0,"jobresult");
   $workunitresult = mysql_result($result,0,"workunitresult");
   $requestid      = mysql_result($result,0,"requestid");
   $iserroneous    = mysql_result($result,0,"iserroneous");
   $errorid        = mysql_result($result,0,"errorid");
   $errormsg       = mysql_result($result,0,"errormsg");
   $errorarg       = mysql_result($result,0,"errorarg");
   $nodeid         = mysql_result($result,0,"nodeid");
   $nodename       = mysql_result($result,0,"nodename");
   
   echo "   <jobresult>\n";
   echo "      <externalid>$externalid</externalid>\n";
   echo "      <jobid>$jobid</jobid>\n";
   echo "      <jobresult><![CDATA[$job]]></jobresult>";
   echo "      <workunitresult>$workunitresult</workunitresult>\n";
   echo "      <requestid>$requestid</requestid>\n";
   echo "      <iserroneous>$iserroneous</iserroneous>\n";
   echo "      <errorid>$errorid</errorid>\n";
   echo "      <error>$error</error>\n";
   echo "      <errormsg><![CDATA[$errormsg]]</errormsg>\n";
   echo "      <errorarg><![CDATA[$errorarg]]</errorarg>\n";
   echo "      <nodeid>$nodeid</nodeid>\n";
   echo "      <nodename>$nodename</nodename>\n";
   echo "   </jobresult>\n";
   
   $upjobqueuequery  = "UPDATE tbjobqueue SET transmitted=1, transmission_dt=NOW() WHERE id=$requestid"; 
   $upjobqueueresult = mysql_query($upjobqueuequery);
   
   $upjobquery  = "UPDATE tbjob SET delivered=$delivered WHERE id=$job_id"; 
   $upjobresult = mysql_query($upjobquery);
  
  $i++;
}

echo "</jobresults>\n";

mysql_close();
?>