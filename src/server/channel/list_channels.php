<?php 
/*
  This PHP script retrieves a list of channels currently available from this server.
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/

 include('../utils/sql2xml/sql2xml.php');
 include('../utils/sql2xml/xsl.php');
 include('../utils/utils.inc.php'); 
 
 // retrieving parameters
 $xml      = getparam("xml", false); 
 
 if (!$xml) prepare_XSLT();
 echo "<channeltypes>\n";
 
 $level_list = Array("channeltype");
 sql2xml('select distinct c.channame, c.chantype from tbchannel c
          order by channame asc; ', $level_list, 0);
 
 echo "</channeltypes>\n";
 if (!$xml) apply_XSLT("channel");

?>