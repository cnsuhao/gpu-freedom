<?php 
session_start(); 
?>
<html>
<head>
<meta http-equiv="refresh" content="180" />
<title>News</title>
</head>
<body>
<h3>News</h3>
<?php
 include_once('../../../server/utils/mydql2i/mysql2i.class.php');
 include("conf/config.inc.php");
 
 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");

 $query="select * from tbnews order by create_dt desc LIMIT 500;";
 $result = mysql_query($query);
 $num = mysql_num_rows($result);
 
 echo "
 <table border='1'>
 <tr>
 <th>create_dt</th>
 <th>newstitle</th>
 <th>source</th>
 </tr>
 ";
 
 $i=0;
 while ($i<$num) {
    $create_dt = mysql_result($result, $i, 'create_dt');
    $newstitle = mysql_result($result, $i, 'newstitle');
 	$source = mysql_result($result, $i, 'source');
    echo "
    <td>$create_dt</td>
    <td>$newstitle</td>
	<td>$source</td>
    </tr>
    ";
 
    $i++;
 }
 echo "</table>";
 
 mysql_close();

?>
</body>
</html>