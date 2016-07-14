<html>
<head>
<title>Store Run Of River CH</title>
</head>
<body>
<?php
 include("../lib/spiderbook/LIB_parse.php");
 include_once('../../../server/utils/mydql2i/mysql2i.class.php');
 include("conf/config.inc.php");
 
 echo "<h3>Store Run Of River CH</h3>";
 
 echo "<table>";
 echo "<p>Connecting...</p>";
 mysql_connect($dbserver, $username, $password);
 @mysql_select_db($database) or die("Unable to select database");
 
 echo "<p>Parsing...</p>";
 
 libxml_use_internal_errors(true);
 
 $xml = simplexml_load_file("data.xml") or die("Error: Cannot create object");
 //print_r($xml);
 
 $j=0;
 foreach ($xml as $mespar) {
   $j++;
   //echo "Type: " . $mespar['Typ'] . '<br/>';
   //echo "Var: " . $mespar['Var'] . '<br/>';
   $type = $mespar['Typ'];
   
   $i=0;
   foreach($mespar as $child) {  
	 $i++;
	 //echo "Child node: " . $child . "<br>";
	 
	 switch ($i) {
		case 1:
			$name = mysql_real_escape_string($child);
		break;
		case 2:
		    $refdatestr = $child;
			$year=substr($refdatestr, 6, 4);
			$month=substr($refdatestr, 3, 2);
			$day=substr($refdatestr,0,2);
			
			$referencedate=$year."-".$month."-".$day;
			
			//echo "*$referencedate*<br>";
		break;
		case 3:
			$timestamp = $child;
			$hour = substr($timestamp, 0, 2);
			$minute = substr($timestamp,3,2);
			//echo "$hour*$minute<br>";
		break;
		case 4:
			$value = str_replace("'", "", $child);
		break;
	 
	 } // switch
	 
	 $query_delete = "DELETE FROM tbhydro_spot where countrycode='CH' and referencedate='$referencedate'
	                  and hour='$hour' and minute='$minute' and name='$name' and id_type=$type";
	 
	 //echo "*$query_delete*";
	 if (!mysql_query($query_delete)) echo mysql_error();
	 
	 $query="INSERT INTO tbhydro_spot (countrycode, referencedate, hour, minute, name, id_type, value, create_dt) 
          VALUES('CH', '$referencedate', '$hour', '$minute', '$name', $type, $value, NOW());";
 
     //echo "$query<br/>";       
     if (!mysql_query($query)) echo mysql_error();
     
   }
 }

   
 
                             
 mysql_close();
 echo "<p>Over, processed $j records.</p>";
 
?>
</body>
</html>