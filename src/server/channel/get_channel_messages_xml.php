<?php
/*
  This PHP script retrieves the current content of a GPU channel
  
  Source code is under GPL, (c) 2002-2011 the Global Processing Unit Team
  
*/
include("../conf/config.inc.php");
$nodeid   = $_GET['nodeid'];
$chantype = $_GET['chantype'];
$channame = $_GET['channame'];
$lastmsg  = $_GET['lastmsg'];
if (($chantype=="") || ($channame=="") || ($lastmsg=="") || ($nodeid=="")) die('<b>Parameters not defined</b>');

mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("<b>Error: Unable to select database, please check settings in conf/config.inc.php</b>");

$mainquery  = "SELECT * from tbchannel WHERE id>$lastmsg AND channame='$channame' AND chantype='$chantype' AND nodeid<>'$nodeid';"; 
$result=mysql_query($mainquery);
if ($result!="") {
 $num=mysql_numrows($result);
} else $num=0; 


echo "<channel>\n";
$i=0;
while ($i<$num) {
   
   $id                = mysql_result($result,$i,"id");
   $nodeid            = mysql_result($result,$i,"nodeid"); 
   $nodename          = mysql_result($result,$i,"nodename");
   $user              = mysql_result($result,$i,"user");
   $channame          = mysql_result($result,$i,"channame");
   $chantype          = mysql_result($result,$i,"chantype");
   $content           = mysql_result($result,$i,"content");
   $usertime_dt       = mysql_result($result,$i,"usertime_dt");
   $create_dt         = mysql_result($result,$i,"create_dt");
   
   echo "   <msg>\n";
   echo "      <id>$id</id>\n";
   echo "      <content><![CDATA[$content]]></content>";
   echo "      <nodeid>$nodeid</nodeid>\n";
   echo "      <nodename>$nodename</nodename>\n";
   echo "      <user>$user</user>\n";
   echo "      <channame>$channame</channame>\n";
   echo "      <chantype>$chantype</chantype>\n";
   echo "      <usertime_dt>$usertime_dt</usertime_dt>\n";
   echo "      <create_dt>$create_dt</create_dt>\n";
   echo "   </msg>\n";

   $i++;
}
echo "</channel>\n";

mysql_close();
?>