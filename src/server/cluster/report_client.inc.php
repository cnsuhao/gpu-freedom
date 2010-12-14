<?php
$id = report_clientinfo($nodename, $nodeid, $country, $region, $city, $zip, $uptime, $totaluptime,
                        $ip, $localip, $port, $acceptincoming, $cputype, $mhz, $ram, $gigaflops,
						$bits, $os, $longitude, $latitude, $version, $team, $userid, $defaultservername,
						$description) {
						
	include("../conf/config.inc.php");					
	mysql_connect($dbserver, $username, $password);
    @mysql_select_db($database) or die("<b>Error: Unable to select database, please check settings in conf/config.inc.php</b>");					
	$query="SELECT id FROM tbclient WHERE nodeid='$nodeid' LIMIT 1"; 
    $result=mysql_query($query);
    if ($result!="") { $num=mysql_numrows($result); } else { $num=0; } 
    if ($num==0) {
	   // we do an INSERT
       $queryinsert="INSERT INTO tbclient (id, nodename, nodeid, country, region, city, zip, 
	                                       uptime, totaluptime,
										   ip, localip, port, acceptincoming,
										   cputype, mhz, ram, gigaflops, bits, os,
										   longitude, latitude,
										   version, team, userid, defaultservername, 
										   description,
										   create_dt, update_dt)
									VALUES('', '$nodename', '$nodeid', '$country', '$region', '$city', '$zip',
                                            $uptime, $totaluptime,
                                           '$ip', '$localip', '$port', $acceptincoming,
                                           '$cputype', $mhz, $ram, $gigaflops, $bits, '$os',
									        $longitude, $latitude,
                                           '$version', '$team', '$userid', '$defaultservername',
                                           '$description',										   
										   NOW(), NOW()
										   );";
       $resultinsert=mysql_query($queryinsert);
	    
       // at the moment we set $id to -1, later we might retrieve $id with a query
       // if necessary
       $id=-1;	   
	} else {
	  // we do an UPDATE
	  $id=mysql_result($result,0,"id");
	  $queryupdate="UPDATE tbclient SET 
	                nodename='$nodename', country='$country', region='$region', city='$city', zip='$zip',
					uptime=$uptime, totaluptime=$totaluptime,
					ip='$ip', localip='$localip', port='$port', acceptincoming=$acceptincoming,
					cputype='$cputype', mhz=$mhz, ram=$ram, gigaflops=$gigaflops, bits=$bits, os='$os',
					longitude=$longitude, latitude=$latitude,
                    version='$version', team='$team', userid='$userid', defaultservername='$defaultservername',
                    description='$description',					
					update_dt=NOW()
					WHERE id=$id;"; 
      $resultupdate=mysql_query($queryupdate);
	}
	mysql_close();	
    return $id;	
}
?>