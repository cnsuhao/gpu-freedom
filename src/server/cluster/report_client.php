<?php
/*
 This class is touched regularly by GPU clients.
 This class creates entries in TBCLIENT, if they not already exist.
 If the computer entry already exists, information like uptime,
 statistics, external IP, if the node is able to get incoming connections,
 how many free connections it has, country, team are updated.
 
 GPU server stores also a last updated information for each computer
 
 This information is then used in list_clients_xsl.php to compile a list
 of currently active nodes.

*/
include("../utils/utils.inc.php");

$nodename = getparam('nodename', '');
$nodeid   = getparam('nodeid', '');
$country  = getparam('country', '');
$region   = getparam('region', '');
$city     = getparam('city', '');
$zip      = getparam('zip', '');
$uptime   = getparam('uptime', '');
$totaluptime = getparam('totaluptime', '');
$localip     = getparam('localip', '');
$port        = getparam('port', '');
$acceptincoming = getparam('acceptincoming', 0);
$cputype        = getparam('cputype', '');
$mhz            = getparam('mhz', 0);
$ram            = getparam('ram', 0);
$gigaflops      = getparam('gigaflops', 0);
$bits           = getparam('bits', 32);
$os             = getparam('os', '');
$longitude   = getparam('longitude', 0);
$latitude    = getparam('latitude', 0);
$version     = getparam('version', '');
$team        = getparam('team', '');
$userid      = getparam('userid', '');
$description = getparam('description', '');

if ( ($nodeid=="") || ($nodename=="") || ($uptime=="") || ($totaluptime=="") ) exit;
$ip = $_SERVER['REMOTE_ADDR'];

$debug=0;
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
    echo "description=$description\n";       
}

include("report_client.inc.php");

$id = report_clientinfo($nodename, $nodeid, $country, $region, $city, $zip, $uptime, $totaluptime,
                        $ip, $localip, $port, $acceptincoming, $cputype, $mhz, $ram, $gigaflops,
						$bits, $os, $longitude, $latitude, $version, $team, $userid, $description);
					
?>