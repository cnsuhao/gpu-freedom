<?php
/*
  This PHP script retrieves a list of active servers from this node.
  This information is replicated from the superserver to the server at regular intervals.
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/

 include('../utils/sql2xml/sql2xml.php'); 
 include('../utils/sql2xml/xsl.php');
 include('../utils/utils.inc.php');
 include('../utils/constants.inc.php');

 $xml = getparam('xml', false); 
 if (!$xml) ob_start();
 
 echo "<servers>\n"; 
 // TODO: reenable condition selecting only servers which are online
 $level_list = Array("server");
 sql2xml("select id, serverid, servername, serverurl, chatchannel, version, superserver, uptime,  
                 longitude, latitude, activenodes, jobinqueue, create_dt, update_dt
         from tbserver 
         -- where update_dt >= ( curdate() - interval $server_update_interval second ) 
		 order by update_dt desc 
		 limit 0, $max_online_servers_xml;
		", $level_list, 0);
 echo "</servers>\n";
 
 if (!$xml) apply_XSLT("supercluster");
?>