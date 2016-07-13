<html>
<head>
<title>Store Meteo BAFU</title>
</head>
<body>
<?php
 include("../lib/spiderbook/LIB_parse.php");
 include_once('../../../server/utils/mydql2i/mysql2i.class.php');
 include("conf/config.inc.php");
 
 echo "<h3>Store Meteo BAFU</h3>";
 
 echo "<table>";
 echo "<p>Connecting...</p>";
 mysql_connect($dbserver, $username, $password);
 @mysql_select_db($database) or die("Unable to select database");
 
 echo "<p>Parsing...</p>";
 
 $lines = file("data.csv");
 $startrow = 3;
 for ( $i = $startrow; $i < sizeof( $lines ); $i++ ) {
   
   $data = explode("|", $lines[$i]);
   echo "<tr>";
   
   $countrycode="CH";
   
   echo "<td>";
   echo $data[0];
   $id_station=$data[0];
   echo "</td>";
   
   echo "<td>";
   echo $data[1];
   
   $referencedatestr = substr($data[1], 0, 8);
   $referencedate = substr($referencedatestr,0,4)."-".substr($referencedatestr,4,2)."-".substr($referencedatestr,6,2);
   echo " ";
   echo $referencedate;
   echo " ";
   $hour = substr($data[1], 8, 2);
   echo $hour;
   echo " ";
   $minute = substr($data[1], 10, 2);
   echo $minute;
   
   echo "</td>";
   
   /* check if data is already inserted */
   if ($i==$startrow) {
		$query_check="select count(*) from tbmeteo_spot where referencedate='$referencedate' and hour='$hour' and minute='$minute'";
		$result_check=mysql_query($query_check);
		$count=mysql_result($result_check, 0, "count(*)");
		if ($count>0) { 
		    mysql_close();
		    die("<b>We already have this data in the database</b>");
		}
   }
   
   echo "<td>";
   echo $data[2];
   if ($data[2]=="-") $data[2]="NULL";
   $temperature=$data[2];
   echo "</td>";
   
   echo "<td>";
   echo $data[3];
   if ($data[3]=="-") $data[3]="NULL";
   $sun_duration=$data[3];
   echo "</td>";
   
   echo "<td>";
   echo $data[4];
   if ($data[4]=="-") $data[4]="NULL";
   $rain=$data[4];
   echo "</td>";
   
   echo "<td>";
   echo $data[5];
   if ($data[5]=="-") $data[5]="NULL";
   $wind_direction=$data[5];
   echo "</td>";
   
   echo "<td>";
   echo $data[6];
   if ($data[6]=="-") $data[6]="NULL";
   $wind_speed=$data[6];
   echo "</td>";
   
   echo "<td>";
   echo $data[7];
   if ($data[7]=="-") $data[7]="NULL";
   $wind_max=$data[7];
   echo "</td>";
   
   echo "<td>";
   echo $data[8];
   if ($data[8]=="-") $data[8]="NULL";
   $relative_humidity=$data[8];
   echo "</td>";
         
   echo "<td>";
   echo $data[9];
   if ($data[9]=="-") $data[9]="NULL";
   $pressure_QNH = $data[9];
   echo "</td>";
   
   echo "<td>";
   echo $data[10];
   if ($data[10]=="-") $data[10]="NULL";
   $pressure_QFE = $data[10];
   echo "</td>";
   
   echo "<td>";
   echo $data[11];
   if (substr($data[11],0,1)=="-") $data[11]="NULL";
   $pressure_QFF = $data[11];
   echo "</td>";
   
   $query="INSERT INTO tbmeteo_spot (countrycode, referencedate, hour, minute, id_station, temperature, sun_duration, rain, wind_direction, wind_speed, wind_max, relative_humidity, pressure_QNH, pressure_QFE, pressure_QFF, insertdate) 
          VALUES('$countrycode', '$referencedate', '$hour', '$minute', '$id_station', $temperature, $sun_duration, $rain, $wind_direction, $wind_speed, $wind_max, $relative_humidity, $pressure_QNH, $pressure_QFE, $pressure_QFF, NOW());";
 
   echo "<td>$query ";       
   if (!mysql_query($query)) echo mysql_error();
   echo "</td>";
   
   echo "</tr>";
  
 }
 
 echo "</table>";
 
 
                             
 mysql_close();
 echo "<p>Over.</p>";
 
?>
</body>
</html>