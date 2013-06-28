<html>
<head>
<meta http-equiv="refresh" content="180" />
<title>Frequencies</title>
</head>
<body>
<h3>Frequencies (refresh each three minutes)</h3>
<?php
 include_once('utils/openflashchart/open_flash_chart_object.php');
 include("conf/config.inc.php");

 echo "<p><tt>./mainloop.sh to update frequencies</tt></p>";
 
 echo "<table>";
 echo "<tr><td>";
 open_flash_chart_object( 500, 250, $dns_name . '/graph/lastfrequencies.php');
 echo "</td>";
 echo '<td><img src="pictures/pylons.jpg" border=0 /></td>';
 echo "</tr>";
 echo "</table>";
 
 echo "<hr>";

 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");

 $query="select * from frequency order by id desc LIMIT 20;";
 $result = mysql_query($query);
 $num = mysql_numrows($result);
 
 echo "
 <table border='1'>
 <tr>
 <th>id</th>
 <th>frequency (Hz)</th>
 <th>networkdiff (s)</th>
 <th>controlarea</th>
 <th>tso</th>
 <th>create_dt</th>
 <th>create_user</th>
 </tr>
 ";
 
 $i=0;
 while ($i<$num) {
    $id          = mysql_result($result, $i, 'id');
    $create_dt   = mysql_result($result, $i, 'create_dt');
    $create_user = mysql_result($result, $i, 'create_user');
    $frequency   = mysql_result($result, $i, 'frequency');
    $netdiff     = mysql_result($result, $i, 'networkdiff');
    $controlarea = mysql_result($result, $i, 'controlarea');
    $tso         = mysql_result($result, $i, 'tso');
    
    echo "
    <tr>
    <td>$id</td>
    <td>$create_dt</td>
    <td>$frequency</td>
    <td>$netdiff</td>
    <td>$controlarea</td>
    <td>$tso</td>
    <td>$create_user</td>
    </tr>
    ";
 
    $i++;
 }
 echo "</table>";
 
 mysql_close();

?>
</body>
</html>