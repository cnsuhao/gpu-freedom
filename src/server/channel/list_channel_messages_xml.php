<?php 
/*
  This PHP script retrieves the current content of a GPU channel
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/

include('../utils/sql2xml/sql2xml.php'); 

// retrieving parameters
if (isset($_GET['channame'])) $channame = $_GET['channame']; else $channame="";
if (isset($_GET['chantype'])) $chantype = $_GET['chantype']; else $chantype="CHAT";
if (isset($_GET['lastmsg'])) $lastmsg  = $_GET['lastmsg']; else $lastmsg=0;

// validating parameters
if ($channame=="") die('ERROR, parameters not defined. Please define at least channame with a valid channel name!');

// if the $lastmsg is not defined, limit results to the last 40 entries
if ($lastmsg==0) $limitation="LIMIT 0,40"; else $limitation = "";

// xml output
echo "<channel>\n";

$level_list = Array("msg");
sql2xml("select c.id, c.content, c.nodeid, c.nodename, c.user, c.channame, c.chantype, c.usertime_dt, c.create_dt from tbchannel c
         where channame='$channame' and chantype='$chantype' and c.id>$lastmsg
		 order by c.id desc $limitation;
		", $level_list, 0);
		
echo "</channel>\n";
?>
