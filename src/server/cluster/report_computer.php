<?php
/*
 This class is touched regularly by GPU nodes.
 This class creates computer entries, if they not already exist.
 If the computer entry already exists, information like uptime,
 statistics, external IP, if the node is able to get incoming connections,
 how many free connections it has, country, team are collected.
 
 FD server stores also a last updated information for each computer
 
 This information is then used in get_connection_list.php to compile a list
 of nodes likely to be good for another node which still does not have
 any connections

*/
$processor 		 = $_GET['processor'];
$nodeid          = $_GET['nodeid'];
$country   		 = $_GET['country'];
$uptime    		 = $_GET['uptime'];
$totuptime 		 = $_GET['totuptime'];
$ip        		 = $_GET['ip'];
$port            = $_GET['port'];
$acceptincoming  = $_GET['acceptincoming'];
$cputype         = $_GET['cputype'];
$mhz		     = $_GET['mhz'];
$ram             = $_GET['ram'];
$operatingsystem = $_GET['os'];
$freeconn        = $_GET['freeconn'];
$maxconn         = $_GET['maxconn'];
$clientversion   = $_GET['version'];
$team            = $_GET['team'];

$lon             = $_GET['lon'];
$lat             = $_GET['lat'];
if ($lon=="") $lon=0;
if ($lat=="") $lat=0;

// gpu processor
$speed            = $_GET['speed'];
$abarth           = $_GET['abarth'];
$threads          = $_GET['threads'];
$inqueue          = $_GET['inqueue'];
$tdown            = $_GET['tdown'];
$tup              = $_GET['tup'];
$listenip         = $_GET['listenip'];
$terra            = $_GET['terra'];
$crawlo           = $_GET['crawlo'];
if ($crawlo=="") $crawlo=0;
if ($terra=="")  $terra=0;

$ips              = $_GET['ips'];

$ip1        		 = $_GET['ip1'];
$ip2        		 = $_GET['ip2'];
$ip3        		 = $_GET['ip3'];
$ip4        		 = $_GET['ip4'];
$ip5        		 = $_GET['ip5'];
$ip6        		 = $_GET['ip6'];
$ip7        		 = $_GET['ip7'];
$ip8        		 = $_GET['ip8'];
$ip9        		 = $_GET['ip9'];
$ip10        		= $_GET['ip10'];


if ($nodeid=="") exit;

include("utils/utils.inc.php");
include("report_nodeinfo.inc.php");

$processorid = report_nodeinfo($processor, $nodeid, $ip, $port, ""/*region*/, $country, $uptime, $totuptime, $acceptincoming,
                               $cputype, $mhz, $ram, $operatingsystem, $freeconn, $maxconn, $clientversion, $team, $lon, $lat);

if (threads!="")							   
 report_gpu_nodeinfo($processorid, $speed, $abarth, $threads, $inqueue, $tdown, $tup, 
                    $ips, $ip1, $ip2, $ip3, $ip4, $ip5, $ip6, $ip7, $ip8, $ip9, $ip10,
                    $abarth, $listenip, $terra, $crawlo);
							   
?>