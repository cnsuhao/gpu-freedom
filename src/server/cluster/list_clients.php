<html>
<head>
<meta http-equiv="refresh" content="180">
<title>List Computers (refreshs each 3 minutes)</title>
</head>
<body>
<?php
include("head.inc.php");
include("conf/config.inc.php");
include("utils/utils.inc.php");
include("utils/constants.inc.php");

$rights = $_COOKIE["rights"];
$cookie_user_id = $_COOKIE["user_id"];
$user_id = $_GET["userid"];
$team_id = $_GET["teamid"];
$entryoffset = $_COOKIE["entryoffset"];
if ($entryoffset=="") {
   $entryoffset=0;
   js_cookie("entryoffset", 0);
}

$cookie_processor =  $_COOKIE["search_processor"];
$cookie_cputype   =  $_COOKIE["search_cputype"];
$cookie_opsys     =  $_COOKIE["search_opsys"];
$cookie_country   =  $_COOKIE["search_country"];
$cookie_mhzfrom   =  $_COOKIE["search_mhzfrom"];
$cookie_mhzto     =  $_COOKIE["search_mhzto"];
$cookie_ramfrom   =  $_COOKIE["search_ramfrom"];
$cookie_ramto     =  $_COOKIE["search_ramto"];
$cookie_uptimefrom      = $_COOKIE["search_uptimefrom"];
$cookie_uptimeto        = $_COOKIE["search_uptimeto"];
$cookie_totuptimefrom   = $_COOKIE["search_totuptimefrom"];
$cookie_totuptimeto     = $_COOKIE["search_totuptimeto"];
$cookie_fromversion     = $_COOKIE["search_fromversion"];
$cookie_toversion       = $_COOKIE["search_toversion"];
$cookie_onlyonline      = $_COOKIE["search_onlyonline"];

show_user_level();
echo "<h2>List Computers</h2>\n";
include("db/mysql/connect.inc.php");

if ($user_id!="") {
   $mainquery  = "SELECT * from tbprocessor where user_id=$user_id order by updated desc LIMIT $entryoffset, $entriesperpage "; 
   $querycount = "SELECT count(*) from tbprocessor";
} else
if ($team_id!="") {
   $query="SELECT * from tbprocessor where team_id=$team_id order by processor LIMIT $entryoffset, $entriesperpage "; 
} else {
   $filterclause = "WHERE (processor LIKE '%$cookie_processor%') ";
   if ($cookie_cputype!="") {
        $filterclause = "$filterclause AND (cputype LIKE '%$cookie_cputype%') ";
   }
   if ($cookie_opsys!="") {
        $filterclause = "$filterclause AND (operatingsystem LIKE '%$cookie_opsys%') ";
   }
   if ($cookie_country!="") {
        $filterclause = "$filterclause AND (country LIKE '%$cookie_country%') ";
   }
   if ($cookie_mhzfrom!="") {
        $filterclause = "$filterclause AND (mhz >= $cookie_mhzfrom) ";
   }
   if ($cookie_mhzto!="") {
        $filterclause = "$filterclause AND (mhz <= $cookie_mhzto) ";
   }
   if ($cookie_ramfrom!="") {
        $filterclause = "$filterclause AND (ram >= $cookie_ramfrom) ";
   }
   if ($cookie_ramto!="") {
        $filterclause = "$filterclause AND (ram <= $cookie_ramto) ";
   }
   if ($cookie_uptimefrom!="") {
        $filterclause = "$filterclause AND (uptime >= $cookie_uptimefrom) ";
   }
   if ($cookie_uptimeto!="") {
        $filterclause = "$filterclause AND (uptime <= $cookie_uptimeto) ";
   }
   if ($cookie_totuptimefrom!="") {
        $filterclause = "$filterclause AND (totuptime >= $cookie_totuptimefrom) ";
   }
   if ($cookie_totuptimeto!="") {
        $filterclause = "$filterclause AND (totuptime <= $cookie_totuptimeto) ";
   }
   
   $mainquery  = "SELECT * from tbprocessor $filterclause order by updated desc LIMIT $entryoffset, $entriesperpage "; 
   $querycount = "SELECT count(*) from tbprocessor $filterclause ";
}

$resultcount=mysql_query($querycount);
if ($resultcount!="") {
 $nbentries=mysql_result($resultcount,0,'count(*)');
} else $nbentries=0;
 
if ($nbentries==0) {
 mysql_close();
 die ("<b>No computers defined</b>");
}

if ($nbentries>$entriesperpage) {
    if ($entryoffset>0) {
      echo " <a href=\"script_top_page.php\">First</a> ";
      echo " <a href=\"script_previous_page.php\">Previous</a> ";
    }
 
    $lastoffset=$nbentries - ($nbentries % $entriesperpage);
    if ($entryoffset<$lastoffset) {
        echo "<a href=\"script_next_page.php\">Next</a> ";
        echo "<a href=\"script_last_page.php?nbentries=$nbentries\">Last</a> ";
    }
    
}

if (($user_id=="") && ($team_id=="")) {
    echo "<a href=\"search_computers.php\">Search</a> ";
    
    if (($cookie_processor!="") || ($cookie_cputype!="") || ($cookie_opsys!="") || ($cookie_country!="") || ($cookie_mhzfrom!="")
        || ($cookie_mhzto!="") || ($cookie_ramfrom!="") || ($cookie_ramto!="") || ($cookie_uptimefrom!="") || ($cookie_uptimeto!="")
        || ($cookie_totuptimefrom!="") || ($cookie_totuptimeto!="")) {
        echo "<b><a href=\"search_cancel.php\">Cancel</a></b> ";
    }
}

// execute the main query with $query
$query=$mainquery;
include("db/mysql/query.inc.php");
include("db/mysql/numrows.inc.php");


$date = time();

echo "<b>($entryoffset/$nbentries)</b><br>\n";
echo "<p>Blue rows denote computers which are currently online.</p>";
echo "<table border=1>";
echo "<tr><th>id</th> <th>processor</th> <th>cpu type</th> <th>OS</th><th>mhz</th> <th>ram</th>";
echo "    <th>country</th><th>uptime (days)</th><th>totuptime (days)</th><th>accepts</th><th>version</th><th>updated</th></tr>";
$i=0;
while ($i<$num) {  

$id=mysql_result($result,$i,"id");
$processor=mysql_result($result,$i,"processor");           
$description=mysql_result($result,$i,"description");
$cputype=mysql_result($result,$i,"cputype");;
$operatingsystem=mysql_result($result,$i,"operatingsystem");;
$mhz=mysql_result($result,$i,"mhz");
$ram=mysql_result($result,$i,"ram");
$cpus=mysql_result($result,$i,"cpus");
$country=mysql_result($result,$i,"country");
$uptime=mysql_result($result,$i,"uptime");
$totuptime=mysql_result($result,$i,"totuptime");
$accept=mysql_result($result,$i,"acceptincoming");
$clientversion=mysql_result($result,$i,"version");
$updated=mysql_result($result,$i,"updated");

// conversion from mySQL to a PHP date
ereg ("([0-9]{4})-([0-9]{1,2})-([0-9]{1,2}) ([0-9]{2}):([0-9]{2}):([0-9]{2})", $updated, $regs);
$updated_php = mktime ($regs[4],$regs[5],$regs[6],$regs[2],$regs[3],$regs[1]);


if (($date-$updated_php)>$update_interval) {
  echo "<tr>";
} else {  
  echo "<tr BGCOLOR=\"#99CCFF\">";
}
echo "<td>$id</td> <td>$processor</td>";
echo "<td>$cputype</td>";
echo "<td>$operatingsystem</td>";
echo "<td>$mhz</td> <td>$ram</td>";
echo "<td>$country</td><td>$uptime</td><td>$totuptime</td>";
echo "<td>$accept</td><td>$clientversion</td>";
echo "<td>$updated</td>";

echo "<td>";
echo "<a href=\"get_processor_stats_2.php?processor=$processor\">Stats</a> ";
if ((($cookie_user_id==$user_id) || ($rights==3)) && ($user_id!="")) {
	echo "<a href=\"remove_computer.php?id=$id&userid=$user_id\">Remove</a>";
}
echo "</td>";

echo "</tr>\n";
$i++; // $i=$i+1;
}
echo "</table>";

include("db/mysql/close.inc.php");
echo "<p>";
if ($rights>=1)
 echo "<a href=\"list_users.php\">Back to List Users</a> ";
?>
<a href="index.php">Back to Main Menu</a>
</body>
</html>