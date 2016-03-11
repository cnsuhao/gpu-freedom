<?php
/*
  This PHP script retrieves the clients which are currently registered with this server
  and displays it as a HTML page.
  
  Source code is under GPL, (c) 2002-2016 the Global Processing Unit Team
  
*/

 include('../utils/sql2xml/sql2xml.php'); 
 include('../utils/sql2xml/xsl.php');
 include('../utils/utils.inc.php');
 include('../utils/constants.inc.php');
 if (getPHPVersion()>=50500) include_once('../utils/mydql2i/mysql2i.class.php');


 $nodeid = getparam("nodeid", "");
 if ($nodeid=="") $nodeclause = ""; else $nodeclause = "and nodeid<>'$nodeid'";
 
 $xml = getparam('xml', false); 
 if (!$xml) ob_start();
 
 echo "<clients>\n"; 
 // TODO: reenable condition selecting only nodes which are online
 $level_list = Array("client");
 sql2xml("select id, nodeid, nodename, country, region, city, zip, os, version, acceptincoming, 
                gigaflops, ram, mhz, nbcpus, bits, isscreensaver, uptime, totaluptime,
 				longitude, latitude, team, description from tbclient 
		 where (1=1)		
         -- and update_dt >= ( curdate() - interval $client_update_interval second ) 
		 $nodeclause
		 order by update_dt desc 
		 limit 0, $max_online_clients_xml;
		", $level_list, 0);
 echo "</clients>\n";
 
 if (!$xml) apply_XSLT();
?>