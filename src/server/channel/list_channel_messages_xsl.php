<?php 
 include('../utils/sql2xml/sql2xml.php');
 ob_start();
 echo "<DOCUMENT>\n"; 

 sql2xml("select c.id, c.content, c.nodeid, c.nodename, c.user, c.channame, c.chantype, c.usertime_dt, c.create_dt,
         cl.country, cl.longitude, cl.latitude 
         from tbchannel c, tbclient cl 
         where c.nodeid=cl.nodeid 
		 order by c.id desc LIMIT 0,40;"
		 , $level_list, 9);
 
 echo "</DOCUMENT>\n";
 include('../utils/sql2xml/xsl.php') 

 ?>