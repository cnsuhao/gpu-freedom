<html>
<head>
<meta http-equiv="refresh" content="180" />
<title>Finance</title>
</head>
<body>
<h3>Finance (refresh each three minutes)</h3>
<?php
 include_once('utils/openflashchart/open_flash_chart_object.php');
 include("conf/config.inc.php");

 echo "<h2>Finance</h2>";
 
 echo "<p><tt>./mainloop.sh to update prices</tt></p>";
 
 echo "<table>";


   mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");

 $query="select distinct name from tickers order by name asc;";
 $result = mysql_query($query);
 $num = mysql_numrows($result);
 $i=0;
 
 while ($i<$num) {
    
	$ticker = mysql_result($result, $i, "name");
	echo "<tr><td>";
	open_flash_chart_object( 500, 250, $dns_name . "/graph/lastprices.php?ticker=$ticker");
	echo "</td></tr>";
	
	$i++;
 }
 
 mysql_close();
 echo "</table>";
 echo "<hr>";
 
?>
</body>
</html>