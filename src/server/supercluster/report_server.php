<?php
/*
 This class is touched regularly by GPU servers.
 This class creates entries in TBSERVER, if they not already exist.
 If the server entry already exists, information like uptime,
 statistics, external IP are updated.
 
 GPU server stores also a last updated information for each server
 
 This information is then used in list_servers.php to compile a list
 of currently active nodes.

*/
include("../utils/utils.inc.php");

$serverid     = getparam('serverid', '');
$servername   = getparam('servername', '');
$serverurl    = getparam('serverurl', '');
$chatchannel  = getparam('chatchannel', '');
$version      = getparam('version', 0);
$superserver  = getparam('superserver', 0);
$uptime       = getparam('uptime', 0);
$longitude    = getparam('longitude', 0);
$latitude     = getparam('latitude', 0);
$activenodes  = getparam('activenodes', 0);
$jobinqueue   = getparam('jobinqueue', 0);

if ( ($serverid=="") || ($servername=="") || ($uptime=="") ) die("ERROR: please specify at least serverid, servername and uptime");
$ip = $_SERVER['REMOTE_ADDR'];

include("report_server.inc.php");

$id = report_serverinfo($serverid, $servername, $serverurl, $chatchannel, $version, $superserver, $uptime, 
                        $longitude, $latitude, $activenodes, $jobinqueue, $ip);
						
echo "OK\n";
					
?>