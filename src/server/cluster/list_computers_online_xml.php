<?php
include("conf/config.inc.php");
include("utils/utils.inc.php");
include("utils/constants.inc.php");

$accept = $_GET["accept"];

if ($max_online_nodes_xml=="") {
	$max_online_nodes_xml = 500;
}

echo "<nodes>\n";
include("db/mysql/connect.inc.php");

// limited to a maximum of $max_online_nodes_xml
$mainquery  = "SELECT * from tbprocessor "; 
$querycount = "SELECT count(*) from tbprocessor ";

if ($accept != "") {
 // if $accept is set to 1, then only nodes which accept incoming
 // connections are shown
 $mainquery  = "$mainquery where acceptincoming=$accept ";
 $querycount = "$querycount where acceptincoming=$accept ";
}
$mainquery  = "$mainquery order by updated desc LIMIT 0, $max_online_nodes_xml";

$resultcount=mysql_query($querycount);
if ($resultcount!="") {
 $nbentries=mysql_result($resultcount,0,'count(*)');
} else $nbentries=0;
 
if ($nbentries==0) {
 mysql_close();
 die ("</nodes>\n");
}

// execute the main query with $query
$query=$mainquery;
include("db/mysql/query.inc.php");
include("db/mysql/numrows.inc.php");

$date = time();

$i=0;
while ($i<$num) {  

  $id=mysql_result($result,$i,"id");
  $processor=mysql_result($result,$i,"processor");           
  $userid=mysql_result($result,$i,"user_id");
  $nodeid=mysql_result($result,$i,"nodeid");
  $ip=mysql_result($result,$i,"ip");
  $port=mysql_result($result,$i,"port");
  $accept=mysql_result($result,$i,"acceptincoming");
  $updated=mysql_result($result,$i,"updated");
  $description=mysql_result($result,$i,"description");
  $cputype=mysql_result($result,$i,"cputype");;
  $mhz=mysql_result($result,$i,"mhz");
  $ram=mysql_result($result,$i,"ram");
  $cpus=mysql_result($result,$i,"cpus");
  $operatingsystem=mysql_result($result,$i,"operatingsystem");
  $teamid=mysql_result($result,$i,"team_id");
  $uptime=mysql_result($result,$i,"uptime");
  $totuptime=mysql_result($result,$i,"totuptime");
  $zip==mysql_result($result,$i,"zip");
  $city==mysql_result($result,$i,"city");
  $region==mysql_result($result,$i,"region");
  $country=mysql_result($result,$i,"country");
  $geolocation_x=mysql_result($result,$i,"geolocation_x");
  $geolocation_y=mysql_result($result,$i,"geolocation_y");
  $freeconn=mysql_result($result,$i,"freeconn");
  $maxconn=mysql_result($result,$i,"maxconn");
  $clientversion=mysql_result($result,$i,"version");

  // conversion from mySQL to a PHP date
  ereg ("([0-9]{4})-([0-9]{1,2})-([0-9]{1,2}) ([0-9]{2}):([0-9]{2}):([0-9]{2})", $updated, $regs);
  $updated_php = mktime ($regs[4],$regs[5],$regs[6],$regs[2],$regs[3],$regs[1]);


  if (($date-$updated_php)<=$update_interval) {
    echo "<node>\n";
    echo "<id>$id</id>\n";
    echo "<processor>$processor</processor>\n";           
    echo "<userid>$userid</userid>\n";
    echo "<nodeid>$nodeid</nodeid>\n";
    echo "<ip>$ip</ip>\n";
    echo "<port>$port</port>\n";
    echo "<accept>$accept</accept>\n";
    echo "<updated>$updated</updated>\n";
    echo "<description>$description</description>\n";
    echo "<cputype>$cputype</cputype>\n";
    echo "<mhz>$mhz</mhz>\n";
    echo "<ram>$ram</ram>\n";
    echo "<cpus>$cpus</cpus>\n";
    echo "<operatingsystem>$operatingsystem</operatingsystem>\n";
    echo "<teamid>$teamid</teamid>\n";
    echo "<uptime>$uptime</uptime>\n";
    echo "<totuptime>$totuptime</totuptime>\n";
    echo "<zip>$zip</zip>\n";
    echo "<city>$city</city>\n";
    echo "<region>$region</region>\n";
    echo "<country>$country</country>\n";
    echo "<geolocation_x>$geolocation_x</geolocation_x>\n";
    echo "<geolocation_y>$geolocation_y</geolocation_y>\n";
    echo "<freeconn>$freeconn</freeconn>\n";
    echo "<maxconn>$maxconn</maxconn>\n";
    echo "<version>$clientversion</version>\n";
	
	// GPU beginning from version 0.9615 report additional information
	$querygpu="SELECT * FROM tbgpuprocessor WHERE processor_id='$id' LIMIT 1"; 
    $resultgpu=mysql_query($querygpu);
    if ($resultgpu!="") { $numgpu=mysql_numrows($resultgpu); } else { $numgpu=0; } 
    if ($numgpu==1) {
        $abarth      = mysql_result($resultgpu,0,"abarth");
	    $speed       = mysql_result($resultgpu,0,"speed");
	    $crawlo      = mysql_result($resultgpu,0,"crawlo");
	    $terra       = mysql_result($resultgpu,0,"terra");
	    $threads     = mysql_result($resultgpu,0,"threads");
	    $inqueue     = mysql_result($resultgpu,0,"inqueue");
	    $trafficdown = mysql_result($resultgpu,0,"trafficdown");
	    $trafficup   = mysql_result($resultgpu,0,"trafficup");
	    $listenip    = mysql_result($resultgpu,0,"listenip");
	    
		$ips=mysql_result($resultgpu,0,"ips");
	    $ip1=mysql_result($resultgpu,0,"ip1");
	    $ip2=mysql_result($resultgpu,0,"ip2");
	    $ip3=mysql_result($resultgpu,0,"ip3");
	    $ip4=mysql_result($resultgpu,0,"ip4");
	    $ip5=mysql_result($resultgpu,0,"ip5");
	    $ip6=mysql_result($resultgpu,0,"ip6");
	    $ip7=mysql_result($resultgpu,0,"ip7");
	    $ip8=mysql_result($resultgpu,0,"ip8");
	    $ip9=mysql_result($resultgpu,0,"ip9");
	    $ip10=mysql_result($resultgpu,0,"ip10");
	     
		echo "<abarth>$abarth</abarth>\n";
		echo "<speed>$speed</speed>\n";
		echo "<crawlo>$crawlo</crawlo>\n";
		echo "<terra>$terra</terra>\n";
		echo "<threads>$threads</threads>\n";
		echo "<inqueue>$inqueue</inqueue>\n";
		echo "<trafficdown>$trafficdown</trafficdown>\n";
		echo "<trafficup>$trafficup</trafficup>\n";
		echo "<listenip>$listenip</listenip>\n";
		 
        echo "<ips>$ips</ips>\n";
		echo "<ip1>$ip1</ip1>\n";
		echo "<ip2>$ip2</ip2>\n";
		echo "<ip3>$ip3</ip3>\n";
		echo "<ip4>$ip4</ip4>\n";
		echo "<ip5>$ip5</ip5>\n";
		echo "<ip6>$ip6</ip6>\n";
		echo "<ip7>$ip7</ip7>\n";
		echo "<ip8>$ip8</ip8>\n";
		echo "<ip9>$ip9</ip9>\n";
		echo "<ip10>$ip10</ip10>\n";
		
	}
	
    echo "</node>\n";
  } else {  
   $i=$num; // to exit the loop, as all other nodes are offline
            // because they did not report to file_distributor in the
			// update interval
  }
  $i++; // $i=$i+1;
}
echo "</nodes>";

include("db/mysql/close.inc.php");

?>