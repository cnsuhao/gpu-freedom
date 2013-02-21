<?php
function report_serverinfo($serverid, $servername, $serverurl, $chatchannel, $version, $superserver, $uptime, $totaluptime,
                        $longitude, $latitude, $activenodes, $jobinqueue) {
						
	include("../conf/config.inc.php");	
    $debug=1;	
	mysql_connect($dbserver, $username, $password);
    @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");					
	$query="SELECT id FROM tbserver WHERE serverid='$serverid' LIMIT 1"; 
    $result=mysql_query($query);
    if ($result!="") { $num=mysql_numrows($result); } else { $num=0; } 
    if ($num==0) {
	   // we do an INSERT
       $queryinsert="INSERT INTO tbserver (id, serverid, servername, serverurl, chatchannel, version, superserver, uptime, totaluptime,
	                                       longitude, latitude, activenodes, jobinqueue,
										   create_dt, update_dt)
									VALUES('', '$serverid', '$servername', '$serverurl', '$chatchannel', $version, $superserver,
                                            $uptime, $totaluptime, $longitude, $latitude, $activenodes, $jobinqueue,
                                           NOW(), NOW()
										   );";
       $resultinsert=mysql_query($queryinsert);
	   if ($debug==1) echo "Insert statement is: $queryinsert\n";
	    
       // at the moment we set $id to -1, later we might retrieve $id with a query
       // if necessary
       $id=-1;	   
	} else {
	  // we do an UPDATE
	  $id=mysql_result($result,0,"id");
	  $queryupdate="UPDATE tbserver SET 
	                servername='$servername', serverurl='$serverurl', chatchannel='$chatchannel', version=$version, superserver=$superserver,
					uptime=$uptime, totaluptime=$totaluptime,
					longitude=$longitude, latitude=$latitude,
                    activenodes=$activenodes, jobinqueue=$jobinqueue,
					update_dt=NOW()
					WHERE id=$id;"; 
      $resultupdate=mysql_query($queryupdate);
	  if ($debug==1) echo "Update statement is: $queryupdate\n";
    }
	mysql_close();	
    echo "OK";
	return $id;	
}
?>
