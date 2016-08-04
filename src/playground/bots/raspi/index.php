<?php 
session_start(); 
?>
<html>
<head>
<meta http-equiv="refresh" content="180" />
<title>Temperature of Raspberry Pi</title>
</head>
<body>
<h3>Temperature of Raspberry Pi</h3>
<?php
 include_once('../../../server/utils/mydql2i/mysql2i.class.php');
 include_once('utils/openflashchart/open_flash_chart_object.php');
 include("conf/config.inc.php");
 
 open_flash_chart_object( 500, 250, $dns_name . '/graph/lasttemperatures.php');

 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");

 $query="select * from tbtemperature order by insert_dt desc LIMIT 1000;";
 $result = mysql_query($query);
 $num = mysql_num_rows($result);
 
 echo "
 <table border='1'>
 <tr>
 <th>referencedate</th>
 <th>temperature (Celsius)</th>
 </tr>
 ";
 
 $i=0;
 while ($i<$num) {
    $referencedate = mysql_result($result, $i, 'insert_dt');
    $temp          = mysql_result($result, $i, 'temperature_raspi');
 	
	if ($temp<50) {
		echo "<tr BGCOLOR=\"#99CCFF\">"; // blue
	} else
	if ($temp<60) {
		echo "<tr BGCOLOR=\"#33FF00\">"; //green
	} else
	if ($temp<70) {
		echo "<tr BGCOLOR=\"#FDD017\">"; //yellow
	} else {
	    echo "<tr BGCOLOR=\"#E41B17\">"; //red
	}
	
    echo "
    <td>$referencedate</td>
    <td>$temp</td>
    </tr>
    ";
 
    $i++;
 }
 echo "</table>";
 
 mysql_close();

?>
</body>
</html>