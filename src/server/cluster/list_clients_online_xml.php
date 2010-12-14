<?php
/*
  This PHP script reports all clients which are currently subscribed to this  
  GPU server
  
  Source code is under GPL, (c) 2002-2010 the Global Processing Unit Team
  
*/

include("../conf/config.inc.php");
mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("<b>Error: Unable to select database, please check settings in conf/config.inc.php</b>");

if ($max_online_nodes_xml=="") {
	$max_online_nodes_xml = 500;
}

echo "<nodes>\n";

$mainquery  = "SELECT * from tbclient order by updated desc LIMIT 0, $max_online_nodes_xml"; 
$result=mysql_query($mainquery);
if ($result!="") {
 $num=mysql_numrows($result);
} else $num=0; 

$date = time();

$i=0;

while ($i<$num) {  

  $id                = mysql_result($result,$i,"id");
  $nodeid            = mysql_result($result,$i,"nodeid"); 
  $nodename          = mysql_result($result,$i,"nodename");
  $defaultservername = mysql_result($result,$i,"defaultservername");
  $country           = mysql_result($result,$i,"country");
  $region            = mysql_result($result,$i,"region");
  $city              = mysql_result($result,$i,"city");
  $zip               = mysql_result($result,$i,"zip");
  $ip                = mysql_result($result,$i,"ip");
  $port              = mysql_result($result,$i,"port");
  $localip           = mysql_result($result,$i,"localip");
  $os                = mysql_result($result,$i,"os");
  $version           = mysql_result($result,$i,"version");
  $acceptincoming    = mysql_result($result,$i,"acceptincoming");
  $gigaflops         = mysql_result($result,$i,"gigaflops");
  $ram               = mysql_result($result,$i,"ram");
  $mhz               = mysql_result($result,$i,"mhz");
  $nbcpus            = mysql_result($result,$i,"nbcpus");
  $bits              = mysql_result($result,$i,"bits");
  $isscreensaver     = mysql_result($result,$i,"isscreensaver");
  $uptime            = mysql_result($result,$i,"uptime");
  $totaluptime       = mysql_result($result,$i,"totaluptime");
  $longitude         = mysql_result($result,$i,"longitude");
  $latitude          = mysql_result($result,$i,"latitude");    
  $userid            = mysql_result($result,$i,"userid");
  $description       = mysql_result($result,$i,"description");
  $cputype           = mysql_result($result,$i,"cputype");;
  $created           = mysql_result($result,$i,"create_dt");
  $updated           = mysql_result($result,$i,"update_dt");
  
  // conversion from mySQL to a PHP date
  ereg ("([0-9]{4})-([0-9]{1,2})-([0-9]{1,2}) ([0-9]{2}):([0-9]{2}):([0-9]{2})", $updated, $regs);
  $updated_php = mktime ($regs[4],$regs[5],$regs[6],$regs[2],$regs[3],$regs[1]);


  if (($date-$updated_php)<=$update_interval) {
    echo "<node>\n";
    echo "  <id>$id</id>\n";
	echo "  <nodeid>$nodeid</nodeid>\n";
    echo "  <nodename>$nodename</nodename>\n";   
    echo "  <defaultservername>$defaultservername</defaultservername>\n";   	
    echo "  <country>$country</country>\n";
    echo "  <region>$region</region>\n";
    echo "  <city>$city</city>\n";
    echo "  <zip>$zip</zip>\n";
    echo "  <ip>$ip</ip>\n";
    echo "  <port>$port</port>\n";
    echo "  <localip>$localip</localip>\n";
    echo "  <os>$os</os>\n";
    echo "  <version>$version</version>\n";
    echo "  <acceptincoming>$acceptincoming</acceptincoming>\n";
    echo "  <gigaflops>$gigaflops</gigaflops>\n";
    echo "  <mhz>$mhz</mhz>\n";
    echo "  <ram>$ram</ram>\n";
    echo "  <nbcpus>$nbcpus</nbcpus>\n";
    echo "  <bits>$bits</bits>\n";
    echo "  <iscreensaver>$isscreensaver</isscreensaver>\n";
    echo "  <uptime>$uptime</uptime>\n";
    echo "  <totaluptime>$totaluptime</totaluptime>\n";
    echo "  <longitude>$longitude</longitude>\n";
    echo "  <latitude>$latitude</latitude>\n";
	echo "  <userid>$userid</userid>\n";
    echo "  <description>$description</description>\n";
    echo "  <cputype>$cputype</cputype>\n";
    echo "  <create_dt>$created</create_dt>\n";
    echo "  <update_dt>$updated</update_dt>\n";
    echo "</node>\n";
  } else {  
   $i=$num; // to exit the loop, as all other nodes are offline
            // because they did not report to server in the
			// update interval
  }
  $i++; 
}
echo "</nodes>";

mysql_close();

?>