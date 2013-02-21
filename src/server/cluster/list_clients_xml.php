<?php
/*
  This PHP script retrieves the clients which are currently registered with this server
  and returns it as XML to be parsed by clients.
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/
 include('../utils/sql2xml/sql2xml.php'); 
 include('../utils/constants.inc.php');

 echo "<clients>\n";

 // TODO: reenable condition selecting only nodes which are online
 $level_list = Array("client");
 sql2xml("select id, nodeid, nodename, country, region, city, zip, os, version, acceptincoming, 
                 gigaflops, ram, mhz, nbcpus, bits, isscreensaver, uptime, totaluptime,
   				 longitude, latitude, team, description from tbclient 
         -- where update_dt >= ( curdate() - interval $update_interval second ) 
		 order by update_dt desc 
		 limit 0, $max_online_clients_xml;
		", $level_list, 0);
		
 echo "</clients>\n";


?>