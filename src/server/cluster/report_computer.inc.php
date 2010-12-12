<?php
function report_nodeinfo($processor, $nodeid, $ip, $port, $region, $country, $uptime, $totuptime, $acceptincoming,
                         $cputype, $mhz, $ram, $operatingsystem, $freeconn, $maxconn, $clientversion, $team, $lon, $lat) {
  include("conf/config.inc.php");
  include("db/mysql/connect.inc.php");
  
  // 1. We first see if we already know this processor. If $nodeid info is provided we use this first
  $query="SELECT * FROM tbprocessor WHERE nodeid='$nodeid' LIMIT 1"; 
  $result=mysql_query($query);
  if ($result!="") { $num=mysql_numrows($result); } else { $num=0; } 
  if ($num==0) {
     // We did not find anything, let's try with the processor string
	$query="SELECT * FROM tbprocessor WHERE processor='$processor' LIMIT 1"; 
    $result=mysql_query($query);
    if ($result!="") {$num=mysql_numrows($result);} else { $num=0;} 
	if ($num==0) {
		// no record exists in the database, we create one, then
		$queryinsert="INSERT INTO tbprocessor (id, processor, nodeid, cputype, mhz, ram, operatingsystem, ip, port, region, country, uptime, totuptime, acceptincoming, freeconn, maxconn, version, geolocation_x, geolocation_y, updated) VALUES('','$processor','$nodeid','$cputype',$mhz,$ram,'$operatingsystem', '$ip', $port, '$region', '$country', $uptime, $totuptime, $acceptincoming, $freeconn, $maxconn, '$clientversion', $lon, $lat, NOW());";
        $resultinsert=mysql_query($queryinsert);
	    mysql_close();
		
		include("db/mysql/connect.inc.php");
		// we reselect to know the id of the newly inserted row
		$query="SELECT * FROM tbprocessor WHERE nodeid='$nodeid' AND processor='$processor' LIMIT 1"; 
        $result=mysql_query($query);
        $id=mysql_result($result,0,"id");
		mysql_close();
		
		return $id;
	}
  }
  $id=mysql_result($result,0,"id");

  // 2. we check if a team exists
  $team_inserted=0;
  if ($team!="") {
	$queryteam="SELECT * FROM tbteam WHERE name='$team' LIMIT 1"; 
	$resultteam=mysql_query($queryteam);
	if ($resultteam!="") { $numteam=mysql_numrows($resultteam); $team_id=mysql_result($resultteam,0,"id"); } 
	else  {$numteam=0;} 
	if ($numteam==0) {
		$queryteam="INSERT INTO tbteam (id, name) VALUES('','$team');";
        $resultteam=mysql_query($queryteam);
	    $team_inserted=1; // introduced as reselect does not work	
	}
  }
  
  // 3. we update relevant fields with the information we have
  $query="UPDATE tbprocessor SET processor='$processor',nodeid='$nodeid',ip='$ip',port=$port,region='$region',country='$country',uptime=$uptime,totuptime=$totuptime,acceptincoming=$acceptincoming,cputype='$cputype',mhz=$mhz,ram=$ram,operatingsystem='$operatingsystem',freeconn=$freeconn,maxconn=$maxconn,version='$clientversion',geolocation_x=$lon, geolocation_y=$lat, updated=NOW()"; 
  if (($team!="") && ($team_inserted==0)) {
	$query = "$query,team_id=$team_id WHERE id=$id";
  } else $query="$query WHERE id=$id";
  
  $result=mysql_query($query);
  mysql_close();
  
  return $id;
}

function report_gpu_nodeinfo($processorid, $speed, $abarth, $threads, $inqueue, $tdown, $tup, 
                             $ips, $ip1, $ip2, $ip3, $ip4, $ip5, $ip6, $ip7, $ip8, $ip9, $ip10,
                             $abarth, $listenip, $terra, $crawlo) {

  // 1. check if there is a row to be updated
  include("conf/config.inc.php");
  include("db/mysql/connect.inc.php");
  
  $query="SELECT * FROM tbgpuprocessor WHERE processor_id='$processorid' LIMIT 1"; 
  $result=mysql_query($query);
  if ($result!="") { $num=mysql_numrows($result); } else { $num=0; } 
  if ($num==0) {						 
        // 2. no record exists in the database, we create one, then
		$queryinsert="INSERT INTO tbgpuprocessor (id, processor_id, speed, abarth, threads, inqueue, trafficdown, trafficup, ips, ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9, ip10, updated, listenip, terra, crawlo) VALUES('','$processorid', $speed, '$abarth', $threads, $inqueue, $tdown, $tup, $ips, '$ip1', '$ip2', '$ip3', '$ip4', '$ip5', '$ip6', '$ip7', '$ip8', '$ip9', '$ip10', NOW(), '$listenip', $terra, $crawlo);";
        $resultinsert=mysql_query($queryinsert);
	    mysql_close();
		exit;
  }
  
  // 3. we update relevant fields with the information we have
  $query="UPDATE tbgpuprocessor SET speed=$speed, abarth='$abarth', threads=$threads, inqueue=$inqueue, trafficdown=$tdown, trafficup=$tup, ips=$ips, ip1='$ip1', ip2='$ip2', ip3='$ip3', ip4='$ip4', ip5='$ip5', ip6='$ip6', ip7='$ip7', ip8='$ip8', ip9='$ip9', ip10='$ip10', updated=NOW(), listenip='$listenip', terra=$terra, crawlo=$crawlo"; 
  $query="$query WHERE processor_id=$processorid";
  $result=mysql_query($query);
  mysql_close();
							 
}
?>