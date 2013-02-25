<?php

/*
  This PHP script contains the logic to handle calls to superservers, so that superservers are aware of our current online status.
  Please note that any server can be also superserver and can receive calls from other servers. However, only powerful servers 
  with reliable connectivity should get superserver status.
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/

function count_records($table) {
  $querycount = "SELECT count(*) from $table;";
  $resultcount=mysql_query($querycount);
  if ($resultcount!="") {
    $nbrecords=mysql_result($resultcount,0,'count(*)');
  } else $nbrecords=0;
  
  return $nbrecords;
}

function call_superserver_phone($serverurl, $activenodes, $jobinqueue, $my_server_id, $uptime) {
  // a connection needs to be established
  include("../conf/config.inc.php"); 
  include("../utils/constants.inc.php");
  $debug = 1;
  
  if ($debug) echo "Calling $serverurl, $activenodes, $jobinqueue\n";
  
  $timeout = 6;
  $old = ini_set('default_socket_timeout', $timeout);
  
  $url = "http://$serverurl/supercluster/report_server.php?serverid=$my_server_id&servername=$my_server_name&serverurl=$my_server_url&chatchannel=$my_default_chat_channel&version=$server_version&uptime=$uptime&longitude=$my_longitude&latitude=$my_latitude&activenodes=$activenodes&jobinqueue=$jobinqueue";
  if ($debug) echo "URL is $url\n";
  
  $handle = fopen($url, 'r');
  ini_set('default_socket_timeout', $old);
  stream_set_timeout($handle, $timeout);
  stream_set_blocking($handle, 0); 
  fclose($handle);

  if ($debug) echo "Phone call over\n";
}


function report_serverinfo($serverid, $servername, $serverurl, $chatchannel, $version, $superserver, $uptime, $longitude, $latitude, $activenodes, $jobinqueue, $ip) {
  include("../conf/config.inc.php");
  
  $debug=1;	
  mysql_connect($dbserver, $username, $password);
  @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php");					
  $query="SELECT id FROM tbserver WHERE serverid='$serverid' LIMIT 1";  
  $result=mysql_query($query);
  if ($result!="") { $num=mysql_numrows($result); } else { $num=0; } 
  
  if ($num==0) {
	   // we do an INSERT
       $queryinsert="INSERT INTO tbserver (id, serverid, servername, serverurl, chatchannel, version, superserver,
	                                       uptime, activenodes, jobinqueue,
										   longitude, latitude, ip,
										   create_dt, update_dt)
									VALUES('', '$serverid', '$servername', '$serverurl', '$chatchannel', $version, $superserver,
                                            $uptime, $activenodes, $jobinqueue,
											$longitude, $latitude, '$ip',						   
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
					update_dt=NOW()
					WHERE id=$id;"; 
      $resultupdate=mysql_query($queryupdate);
	  if ($debug==1) echo "Update statement is: $queryupdate\n";
    }
	mysql_close();
	return $id;	
}


function call_nearest_superservers_to_report_my_status() {
  include("../conf/config.inc.php"); 
  include("../utils/constants.inc.php");
  include_once("../utils/parameters.inc.php");

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
  echo "$query\n";		  
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
  include("../utils/parameters.inc.php");  
  
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
  echo "$query\n";		  
  $result=mysql_query($query);
  
  if ($result!="") { $num=mysql_numrows($result); } else { $num=0; }
  
  
  mysql_close();

/*
$oDOM = new DOMDocument();
$oDOM->loadXML(file_get_contents('books.xml')); #See: http://msdn.microsoft.com/en-us/library/ms762271(VS.85).aspx
foreach ($oDOM->getElementsByTagName('book') as $oBookNode)
{
    printf(
        "INSERT INTO table (title, author, description) VALUES ('%s', '%s', '%s')",
        mysql_real_escape_string($oBookNode->getElementsByTagName('title')->item(0)->nodeValue),
        mysql_real_escape_string($oBookNode->getElementsByTagName('author')->item(0)->nodeValue),
        mysql_real_escape_string($oBookNode->getElementsByTagName('description')->item(0)->nodeValue)
    );
}
*/
}

function call_superserver_if_required() {
  include("../conf/config.inc.php"); 
  include("../utils/constants.inc.php");
  include("../utils/parameters.inc.php");

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