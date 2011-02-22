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

$debug=1;
if ($debug==1) {
	echo "nodename=$nodename\n";
    echo "nodeid=$nodeid\n";
    echo "country=$country\n";
    echo "region=$region\n";
    echo "city=$city\n";
    echo "zip=$zip\n";   	
    echo "uptime=$uptime\n";    	
    echo "totaluptime=$totaluptime\n"; 
    echo "ip=$ip\n";        		
    echo "localip=$localip\n";       
    echo "port=$port\n";           
    echo "acceptincoming=$acceptincoming\n"; 
    echo "cputype=$cputype\n";        
    echo "mhz=$mhz\n";		   
    echo "ram=$ram\n";            
    echo "gigaflops=$gigaflops\n";   
    echo "bits=$bits\n";        
    echo "os=$os\n";            
    echo "longitude=$longitude\n";     
    echo "latitude=$latitude\n";    
    echo "version=$version\n";   
    echo "team=$team\n";         
    echo "userid=$userid\n";        
    echo "defaultservername=$defaultservername\n";
    echo "description=$description\n";       
}

include("report_client.inc.php");

$id = report_clientinfo($nodename, $nodeid, $country, $region, $city, $zip, $uptime, $totaluptime,
                        $ip, $localip, $port, $acceptincoming, $cputype, $mhz, $ram, $gigaflops,
						$bits, $os, $longitude, $latitude, $version, $team, $userid, $defaultservername,
						$description);

echo "Done!\n";						
?>