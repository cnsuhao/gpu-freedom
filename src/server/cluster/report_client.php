<?php
/*
 This class is touched regularly by GPU clients.
 This class creates entries in TBCLIENT, if they not already exist.
 If the computer entry already exists, information like uptime,
 statistics, external IP, if the node is able to get incoming connections,
 how many free connections it has, country, team are updated.
 
 GPU server stores also a last updated information for each computer
 
 This information is then used in list_computers_online_xml.php to compile a list
 of currently active nodes.

*/

$nodename 		 = $_GET['nodename'];
$nodeid          = $_GET['nodeid'];
$country   		 = $_GET['country'];
$region   		 = $_GET['region'];
$city   		 = $_GET['city'];
$zip   		     = $_GET['zip'];
$uptime    		 = $_GET['uptime'];
$totaluptime     = $_GET['totaluptime'];
$ip        		 = $_GET['ip'];
$localip         = $_GET['localip'];
$port            = $_GET['port'];
$acceptincoming  = $_GET['acceptincoming'];
$cputype         = $_GET['cputype'];
$mhz		     = $_GET['mhz'];
$ram             = $_GET['ram'];
$gigaflops       = $_GET['gigaflops'];
$bits            = $_GET['bits'];
$os              = $_GET['os'];
$longitude       = $_GET['longitude'];
$latitude        = $_GET['latitude'];
$version         = $_GET['version'];
$team            = $_GET['team'];
$userid          = $_GET['userid'];
$defaultservername = $_GET['defaultservername'];
$description       = $_GET['description'];

if ($nodeid=="") exit;

include("report_client.inc.php");

$id = report_clientinfo($nodename, $nodeid, $country, $region, $city, $zip, $uptime, $totaluptime,
                        $ip, $localip, $port, $acceptincoming, $cputype, $mhz, $ram, $gigaflops,
						$bits, $os, $longitude, $latitude, $version, $team, $userid, $defaultservername,
						$description);
							   
?>