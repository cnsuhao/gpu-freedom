<?php 
session_start(); 
?>
<html>
<head>
<meta http-equiv="refresh" content="180" />
<title>Meteo</title>
</head>
<body>
<h3>Meteo (refresh each three minutes)</h3>
<?php
 include("conf/config.inc.php");
 include_once('../../../server/utils/mydql2i/mysql2i.class.php');

 echo "<p><tt>./mainloop.sh to update meteo data</tt></p>";
 echo "<hr>";
 
 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");

 $query="select * from tbmeteo_spot order by referencedate desc, hour desc, minute desc, id_station LIMIT 5000;";
 $result = mysql_query($query);
 $num = mysql_num_rows($result);
 
 echo "
 <table border='1'>
 <tr>
 <th>countrycode</th>
 <th>referencedate</th>
 <th>hour</th>
 <th>minute</th>
 <th>id_station</th>
 <th>temperature</th>
 <th>sun_duration</th>
 <th>rain</th>
 <th>wind_direction</th>
 <th>wind_speed</th>
 <th>wind_max</th>
 <th>relative_humidity</th>
 <th>pressure_QNH</th>
 <th>pressure_QFE</th>
 <th>pressure_QFF</th>
 <th>insertdate</th>
 </tr>
 ";
 
 $i=0;
 while ($i<$num) {
    $countrycode   = mysql_result($result, $i, 'countrycode');
    $referencedate = mysql_result($result, $i, 'referencedate');
    $hour          = mysql_result($result, $i, 'hour');
    $minute        = mysql_result($result, $i, 'minute');
    $id_station    = mysql_result($result, $i, 'id_station');
    $temperature   = mysql_result($result, $i, 'temperature');
    $sun_duration  = mysql_result($result, $i, 'sun_duration');
    $rain          = mysql_result($result, $i, 'rain');
    $wind_direction= mysql_result($result, $i, 'wind_direction');
    $wind_speed        = mysql_result($result, $i, 'wind_speed');
    $wind_max          = mysql_result($result, $i, 'wind_max');
    $relative_humidity = mysql_result($result, $i, 'relative_humidity');
    $pressure_QNH      = mysql_result($result, $i, 'pressure_QNH');
    $pressure_QFE      = mysql_result($result, $i, 'pressure_QFE');
    $pressure_QFF      = mysql_result($result, $i, 'pressure_QFF');
    $insertdate        = mysql_result($result, $i, 'insertdate');
    
	
	
    echo "
    <tr>
    <td>$countrycode</td>
    <td>$referencedate</td>
    <td>$hour</td>
    <td>$minute</td>
    <td>$id_station</td>
    <td>$temperature</td>
    <td>$sun_duration</td>
    <td>$rain</td>
    <td>$wind_direction</td>
    <td>$wind_speed</td>
    <td>$wind_max</td>
    <td>$relative_humidity</td>
    <td>$pressure_QNH</td>
    <td>$pressure_QFE</td>
    <td>$pressure_QFF</td>
    <td>$insertdate</td>
    </tr>
    ";
 
    $i++;
 }
 echo "</table>";
 
 mysql_close();

?>
</body>
</html>