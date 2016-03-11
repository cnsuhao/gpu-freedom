<?php

/*
  This PHP script contains the logic to handle calls to superservers, so that superservers are aware of our current online status.
  It also contains logic to retrieve the server list from a superserver and to load it in our database.
  
  Please note that any server can be also superserver and can receive calls from other servers. However, only powerful servers 
  with reliable connectivity should get superserver status.
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/

include_once("../utils/parameters.inc.php");
include_once("../utils/urls.inc.php");
include_once("../utils/utils.inc.php");
if (getPHPVersion()>=50500) include_once('../utils/mydql2i/mysql2i.class.php');


function count_records($table) {
  $querycount = "SELECT count(*) from $table;";
  $resultcount=mysql_query($querycount);
  if ($resultcount!="") {
    $nbrecords=mysql_result($resultcount,0,'count(*)');
  } else $nbrecords=0;
  
  return $nbrecords;
}

function report_serverinfo($serverid, $servername, $serverurl, $chatchannel, $version, $superserver, $uptime, $longitude, $latitude, $activenodes, $jobinqueue, $ip) {
  include("../conf/config.inc.php");
  
  $debug=0;	
  mysql_connect($dbserver, $username, $password);
  @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");					
  $query="SELECT id FROM tbserver WHERE serverid='$serverid' LIMIT 1";  
  $result=mysql_query($query);
  if ($result!="") { $num=mysql_numrows($result); } else { $num=0; } 
  
  if ($num==0) {
	   // we do an INSERT
       $queryinsert="INSERT INTO tbserver (id, serverid, servername, serverurl, chatchannel, version, superserver,
	                                       uptime, activenodes, jobinqueue,
										   longitude, latitude, ip, pos,
										   create_dt, update_dt)
									VALUES('', '$serverid', '$servername', '$serverurl', '$chatchannel', $version, $superserver,
                                            $uptime, $activenodes, $jobinqueue,
											$longitude, $latitude, '$ip', POINT($longitude , $latitude) 						   
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
	                servername='$servername', serverurl='$serverurl', chatchannel='$chatchannel', superserver=$superserver,
					uptime=$uptime, activenodes=$activenodes, jobinqueue=$jobinqueue,
					ip='$ip', 
					longitude=$longitude, latitude=$latitude, 
                    version=$version, 	
                    pos=POINT($longitude , $latitude),					
					update_dt=NOW()
					WHERE id=$id;"; 
      $resultupdate=mysql_query($queryupdate);
	  if ($debug==1) echo "Update statement is: $queryupdate\n";
    }
	mysql_close();
	
	return $id;	
}

function call_superserver_phone($serverurl, $activenodes, $jobinqueue, $my_server_id, $uptime) {
  // a connection needs to be established
  include("../conf/config.inc.php"); 
  include("../utils/constants.inc.php");
  $debug = 0;
  
  if ($debug) echo "Calling $serverurl, $activenodes, $jobinqueue\n";
  
  $serverid   = urlencode($my_server_id);
  $servername = urlencode($my_server_name);
  $myurl      = urlencode($my_server_url);
  $chat       = urlencode($my_default_chat_channel);
  $version    = urlencode($server_version);
  $up         = urlencode($uptime);
  $lon        = urlencode($my_longitude);
  $lat        = urlencode($my_latitude);
  
  $url = "http://$serverurl/supercluster/report_server.php?serverid=$serverid&servername=$servername&serverurl=$myurl&chatchannel=$chat&version=$version&uptime=$up&longitude=$lon&latitude=$lat&activenodes=$activenodes&jobinqueue=$jobinqueue";
  
  if ($debug) echo "URL is $url\n"; 
  touch_url($url, $max_superserver_timeout);
  if ($debug) echo "Phone call over\n";
}


function call_nearest_superservers_to_report_my_status() {
  include("../conf/config.inc.php"); 
  include("../utils/constants.inc.php");

  mysql_connect($dbserver, $username, $password);
  @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");					
  
  // collect data for the phone call
  $activenodes  = count_records("tbclient");
  $jobinqueue   = count_records("tbjobqueue");
  
  $my_server_id = get_db_parameter("CONFIGURATION", "SERVER_ID", "missing");
  if ($my_server_id=="missing") die("ERROR: Internal server error, SERVER_ID in TBPARAMETER is missing!");  
  
  $uptime = get_db_parameter("TIME", "UPTIME", "missing");
  if ($uptime=="missing") die("ERROR: Internal error, UPTIME parameter is missing");
   
  // retrieve servers which are eligible to get the phone call
  $query="SELECT serverid, serverurl, longitude, latitude, PLANAR_DISTANCE(latitude, longitude, $my_longitude, $my_latitude) as distance
          FROM tbserver 
          WHERE serverid<>'$my_server_id' 
		  AND superserver=1
		  ORDER BY distance ASC
          LIMIT $nb_superserver_informed;";
  //echo "$query\n";		  
  $result=mysql_query($query);
  
  if ($result!="") { $num=mysql_numrows($result); } else { $num=0; }
  
  // do the calls
  $i = 0;
  while ($i<$num) {
	$serverurl = mysql_result($result, $i, 'serverurl'); 
	call_superserver_phone($serverurl, $activenodes, $jobinqueue, $my_server_id, $uptime);
	$i++;
  } 
  mysql_close();
  
  // now we upgrade also this information on ourself in our tbserver table
  report_serverinfo($my_server_id, $my_server_name, $my_server_url, $my_default_chat_channel, $server_version, $am_i_superserver, $uptime, 
                    $my_longitude, $my_latitude, $activenodes, $jobinqueue, "localhost");
  echo "OK";
}


function retrieve_server_list_from_nearest_superserver() {
  include("../conf/config.inc.php");
  include("../utils/constants.inc.php");
  
  mysql_connect($dbserver, $username, $password);
  @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");			

  $my_server_id = get_db_parameter("CONFIGURATION", "SERVER_ID", "missing");
  if ($my_server_id=="missing") die("ERROR: Internal server error, SERVER_ID in TBPARAMETER is missing!");  
  
  $query="SELECT servername, serverurl, longitude, latitude, PLANAR_DISTANCE(latitude, longitude, $my_longitude, $my_latitude) as distance
          FROM tbserver 
          WHERE serverid<>'$my_server_id' 
		  AND superserver=1
		  ORDER BY distance ASC
          LIMIT 1";
  //echo "$query\n";		  
  $result = mysql_query($query);
  if ($result!="") { $num=mysql_numrows($result); } else { $num=0; }
  
  if ($num>0) $url = mysql_result($result, 0, 'serverurl');
  
  mysql_close();

  $suffix = create_unique_id();
  $filename = "../temp/servers_$suffix.xml";
  $myid = urlencode($my_server_id);
  save_url("$url/supercluster/list_servers.php?xml=1&serverid=$myid", $filename, $max_superserver_timeout);
  
  $oDOM = new DOMDocument();
  $oDOM->loadXML(file_get_contents($filename)); #See: http://msdn.microsoft.com/en-us/library/ms762271(VS.85).aspx
  
  foreach ($oDOM->getElementsByTagName('server') as $oServerNode)   {
        $serverid   = mysql_real_escape_string($oServerNode->getElementsByTagName('serverid')->item(0)->nodeValue);
		if ($serverid==$my_server_id) continue; // we do not update information on our server from unreliable external sources :-)
		                                        // a superserver should not provide information on ourself, as we pass our server id
        $servername = mysql_real_escape_string($oServerNode->getElementsByTagName('servername')->item(0)->nodeValue);
        $serverurl  = mysql_real_escape_string($oServerNode->getElementsByTagName('serverurl')->item(0)->nodeValue); 
        $chatchannel = mysql_real_escape_string($oServerNode->getElementsByTagName('chatchannel')->item(0)->nodeValue); 
        $version = mysql_real_escape_string($oServerNode->getElementsByTagName('version')->item(0)->nodeValue); 
        $superserver = mysql_real_escape_string($oServerNode->getElementsByTagName('superserver')->item(0)->nodeValue); 
        $uptime = mysql_real_escape_string($oServerNode->getElementsByTagName('uptime')->item(0)->nodeValue); 
        $longitude = mysql_real_escape_string($oServerNode->getElementsByTagName('longitude')->item(0)->nodeValue); 
        $latitude = mysql_real_escape_string($oServerNode->getElementsByTagName('latitude')->item(0)->nodeValue); 
        $activenodes = mysql_real_escape_string($oServerNode->getElementsByTagName('activenodes')->item(0)->nodeValue); 
        $jobinqueue = mysql_real_escape_string($oServerNode->getElementsByTagName('jobinqueue')->item(0)->nodeValue); 	

        report_serverinfo($serverid, $servername, $serverurl, $chatchannel, $version, $superserver, $uptime, $longitude, $latitude, $activenodes, $jobinqueue, "");		
  } //for each
  
  unlink($filename);
}

function call_superserver_if_required() {
  include("../conf/config.inc.php"); 
  include("../utils/constants.inc.php");

  mysql_connect($dbserver, $username, $password);
  @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");	

  $last_update = get_db_parameter("TIME", "LAST_SUPERSERVER_CALL", "missing");
  if ($last_update=="missing") die("ERROR: Internal error, LAST_SUPERSERVER_CALL parameter is missing");
  
  mysql_close();
  
  $current_time = time();
  $difference = ($current_time-$last_update);
  
  if ( $difference > $server_update_interval ) {
            mysql_connect($dbserver, $username, $password);
            @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");	
  
			set_db_parameter("TIME", "LAST_SUPERSERVER_CALL", $current_time);
			$uptime = get_db_parameter("TIME", "UPTIME", "missing");
			if ($uptime=="missing") die("ERROR: Internal error, UPTIME parameter is missing");
			
			if ($difference <= 24 * 3600) {  // if there is more than one day of difference, we assume the server was offline during this time

				$new_uptime = $uptime + $difference;
				set_db_parameter("TIME", "UPTIME", $new_uptime);
			}
			
			mysql_close();
			
			call_nearest_superservers_to_report_my_status();
			retrieve_server_list_from_nearest_superserver();
  }
  
}


?>