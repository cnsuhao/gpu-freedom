<?php 
/*
  This PHP script retrieves the current content of a GPU channel,
  including some client information.
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/
 include('../utils/sql2xml/sql2xml.php');
 include('../utils/sql2xml/xsl.php'); 
 include('../utils/utils.inc.php'); 
 
 // retrieving parameters
 $xml      = getparam("xml", false);
 $channame = getparam("channame","%");
 $chantype = getparam("chantype","CHAT");
 $lastmsg  = getparam("lastmsg", 0);
 
 // if the $lastmsg is not defined, limit results to the last 40 entries
 if (($lastmsg==0) || ($channame=="")) $limitation="LIMIT 0,40"; else $limitation = "";

 if (!$xml) prepare_XSLT();
 echo "<channel>\n"; 
 $level_list = Array("msg", "client");
 sql2xml("select c.id, c.content, c.nodeid, c.nodename, c.user, c.channame, c.chantype, c.usertime_dt, c.create_dt,
         cl.country, cl.longitude, cl.latitude 
         from tbchannel c, tbclient cl 
         where c.nodeid=cl.nodeid and channame like '$channame' and chantype='$chantype' and c.id>$lastmsg
		 order by c.id desc $limitation;"
		 , $level_list, 9);
 
 echo "</channel>\n";
 if (!$xml) apply_XSLT();
 
 ?>