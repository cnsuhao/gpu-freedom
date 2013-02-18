<html>
<head>
<meta http-equiv="refresh" content="60">
<title>GPU Server - List latest channel messages (refreshes each minute)</title>
</head>
<body>
<img src="../images/gpu-inverse.jpg" border="0"><br>
<h2>List latest channel messages</h2>
<?php
/*
  This PHP script retrieves the current content of a GPU channel
  
  Source code is under GPL, (c) 2002-2011 the Global Processing Unit Team
  
*/
include("../conf/config.inc.php");
mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("<b>Error: Unable to select database, please check settings in conf/config.inc.php</b>");

$mainquery  = "SELECT * from tbchannel ORDER BY id DESC LIMIT 0,40;"; 
$result=mysql_query($mainquery);
if ($result!="") {
 $num=mysql_numrows($result);
} else $num=0; 


echo "<table border=1>\n";
echo "<tr>
      <th>id</th>
	  <th>channel type</th>
	  <th>channel name</th>
      <th>nodename</th>
      <th>user</th>
	  <th>content</th>
	  <th>create datum</th>
	  </tr>\n
	  ";
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
   
   echo "   <tr>\n";
   echo "      <td>$id</td>\n";
   echo "      <td>$chantype</td>\n";
   echo "      <td>$channame</td>\n";
   echo "      <td>$nodename</td>\n";
   echo "      <td>$user</td>\n";
   echo "      <td>$content</td>";
   echo "      <td>$create_dt</td>\n";
   echo "   </tr>\n";

   $i++;
}
echo "</table>\n";

mysql_close();
?>
</body>
</html>