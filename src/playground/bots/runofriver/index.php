<?php 
session_start(); 
?>
<html>
<head>
<meta http-equiv="refresh" content="180" />
<title>Run of River</title>
</head>
<body>
<h3>Run Of River (refresh each three minutes)</h3>
<?php
 include("conf/config.inc.php");
 include_once('../../../server/utils/mydql2i/mysql2i.class.php');

 echo "<p><tt>./mainloop.sh to update run of river data</tt></p>";
 echo "<hr>";
 
 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");

 $query="select * from tbhydro_spot order by create_dt desc, name LIMIT 5000;";
 $result = mysql_query($query);
 $num = mysql_num_rows($result);
 
 echo "
 <table border='1'>
 <tr>
 <th>countrycode</th>
 <th>referencedate</th>
 <th>hour</th>
 <th>minute</th>
 <th>name</th>
 <th>type</th>
 <th>value</th>
 <th>create_dt</th>
 </tr>
 ";
 
 $i=0;
 while ($i<$num) {
    $countrycode   = mysql_result($result, $i, 'countrycode');
    $referencedate = mysql_result($result, $i, 'referencedate');
    $hour          = mysql_result($result, $i, 'hour');
    $minute        = mysql_result($result, $i, 'minute');
    $name          = mysql_result($result, $i, 'name');
    $type          = mysql_result($result, $i, 'type');
    $value         = mysql_result($result, $i, 'value');
    $create_dt     = mysql_result($result, $i, 'create_dt');
 	
	
    echo "
    <tr>
    <td>$countrycode</td>
    <td>$referencedate</td>
    <td>$hour</td>
    <td>$minute</td>
    <td>$name</td>
    <td>$type</td>
    <td>$value</td>
    <td>$create_dt</td>
    </tr>
    ";
 
    $i++;
 }
 echo "</table>";
 
 mysql_close();

?>
</body>
</html>