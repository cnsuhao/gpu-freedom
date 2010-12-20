<html>
<head>
<meta http-equiv="refresh" content="180">
<title>GPU Server - List Clients (refreshs each 3 minutes)</title>
</head>
<body>
<img src="../images/gpu-inverse.jpg" border="0"><br>
<?php
include("../conf/config.inc.php");
include("../utils/utils.inc.php");

$entryoffset = $_COOKIE["entryoffset"];
if ($entryoffset=="") {
   $entryoffset=0;
   js_cookie("entryoffset", 0);
}

$cookie_nodename  =  $_COOKIE["search_nodename"];
$cookie_cputype   =  $_COOKIE["search_cputype"];
$cookie_os        =  $_COOKIE["search_os"];
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

$filterclause = "WHERE (nodename LIKE '%$cookie_nodename%') ";
if ($cookie_cputype!="") {
        $filterclause = "$filterclause AND (cputype LIKE '%$cookie_cputype%') ";
}
if ($cookie_opsys!="") {
        $filterclause = "$filterclause AND (os LIKE '%$cookie_os%') ";
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
        $filterclause = "$filterclause AND (totaluptime >= $cookie_totuptimefrom) ";
}
if ($cookie_totuptimeto!="") {
        $filterclause = "$filterclause AND (totaluptime <= $cookie_totuptimeto) ";
}

if (($cookie_nodename!="") || ($cookie_cputype!="") || ($cookie_os!="") || ($cookie_country!="") || ($cookie_mhzfrom!="")
        || ($cookie_mhzto!="") || ($cookie_ramfrom!="") || ($cookie_ramto!="") || ($cookie_uptimefrom!="") || ($cookie_uptimeto!="")
        || ($cookie_totuptimefrom!="") || ($cookie_totuptimeto!="")) {
        echo "<b><a href=\"search_cancel.php\">Cancel</a></b> ";
}
   
$mainquery  = "SELECT * from tbclient $filterclause order by update_dt desc LIMIT $entryoffset, $entriesperpage "; 
$querycount = "SELECT count(*) from tbclient $filterclause ";

mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("<b>Error: Unable to select database, please check settings in conf/config.inc.php</b>");

$resultcount=mysql_query($querycount);
if ($resultcount!="") {
 $nbentries=mysql_result($resultcount,0,'count(*)');
} else $nbentries=0;
 
if ($nbentries==0) {
 mysql_close();
 die ("<b>No clients reported.</b>");
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

echo "<a href=\"search_clients.php\">Search</a> ";
    
$result= mysql_query($mainquery);
$num   = mysql_numrows($result);
$date  = time();

echo "<b>($entryoffset/$nbentries)</b><br>\n";
echo "<p>Blue rows denote computers which are currently online.</p>";
echo "<table border=1>";
echo "<tr><th>id</th> <th>name</th> <th>cpu type</th> <th>OS</th><th>mhz</th> <th>ram</th>";
echo "    <th>country</th><th>uptime (days)</th><th>totaluptime (days)</th><th>version</th><th>updated</th></tr>";

$i=0;

while ($i<$num) {  

  $id=mysql_result($result,$i,"id");
  $nodename=mysql_result($result,$i,"nodename");           
  $cputype=mysql_result($result,$i,"cputype");;
  $os=mysql_result($result,$i,"os");;
  $mhz=mysql_result($result,$i,"mhz");
  $ram=mysql_result($result,$i,"ram");
  $nbcpus=mysql_result($result,$i,"nbcpus");
  $country=mysql_result($result,$i,"country");
  $uptime=mysql_result($result,$i,"uptime");
  $totaluptime=mysql_result($result,$i,"totaluptime");
  $version=mysql_result($result,$i,"version");
  $updated=mysql_result($result,$i,"update_dt");

  // conversion from mySQL to a PHP date
  ereg ("([0-9]{4})-([0-9]{1,2})-([0-9]{1,2}) ([0-9]{2}):([0-9]{2}):([0-9]{2})", $updated, $regs);
  $updated_php = mktime ($regs[4],$regs[5],$regs[6],$regs[2],$regs[3],$regs[1]);


  if (($date-$updated_php)>$update_interval) {
    echo "<tr>";
  } else {  
    echo "<tr BGCOLOR=\"#99CCFF\">";
  }
  
  echo "<td>$id</td> <td>$nodename</td>";
  echo "<td>$cputype</td>";
  echo "<td>$os</td>";
  echo "<td>$mhz</td> <td>$ram</td>";
  echo "<td>$country</td><td>$uptime</td><td>$totaluptime</td>";
  echo "<td>$version</td><td>$updated</td>";
  echo "</tr>\n";
  
  $i++; // $i=$i+1;
}
echo "</table>";

mysql_close();
?>
</body>
</html>