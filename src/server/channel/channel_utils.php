<?php
/*
  Retrieves the latest msg id for a given channel, substracting some offset
*/
function retrieve_valid_lastmsg($channame, $chantype) {
	include("../conf/config.inc.php");
    
	$db_cn = mysql_connect($dbserver, $username, $password);
	@mysql_select_db($database, $db_cn) or die("<b>Error: Unable to select database, please check settings in conf/config.inc.php</b>");
	
	$sql    = "select max(id) from tbchannel where channame='$channame' and chantype='$chantype';";
	$result = mysql_query($sql, $db_cn);
	$lastmsg = mysql_result($result, 0, "max(id)");
	
	mysql_close();
	
	return $lastmsg - $channel_entries_offset;
}


?>