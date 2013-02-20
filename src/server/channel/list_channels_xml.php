<?php 
/*
  This PHP script retrieves a list of channels currently available from this server.
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/

include('../utils/sql2xml/sql2xml.php'); 
 
echo "<channeltype>\n";

$level_list = Array("type");
sql2xml('select distinct c.channame, c.chantype from tbchannel c', $level_list, 0);

echo "</channeltype>\n";

?>