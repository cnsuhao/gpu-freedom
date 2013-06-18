<html>
<head>
<meta http-equiv="refresh" content="180" />
<title>Buffer Arbitrage</title>
</head>
<body>
<h3>Buffer Arbitrage (refresh each three minutes)</h3>
<?php
 include_once('utils/openflashchart/open_flash_chart_object.php');
 include("conf/config.inc.php");

 echo "<p><tt>./mainloop.sh to update prices</tt></p>";
 
 echo "<table>";
 echo "<tr><td>";
 open_flash_chart_object( 500, 250, $dns_name . '/graph/lastprices.php');
 echo "</td><td>";
 open_flash_chart_object( 500, 250, $dns_name . '/graph/lastvolumes.php');
 echo "</td>";
 echo '<td><img src="pictures/linuxbc.jpg" border=0 /></td>';
 echo "</tr>";
 echo "</table>";
 
 echo "<hr>";
 
 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");

 $query="select * from pricevalue order by id desc LIMIT 20;";
 $result = mysql_query($query);
 $num = mysql_numrows($result);
 
 echo "
 <table border='1'>
 <tr>
 <th>id</th>
 <th>create_dt</th>
 <th>price</th>
 <th>low</th>
 <th>high</th>
 <th>volume</th>
 <th>my avg</th>
 <th>avg exchange</th>
 <th>delta avg</th>
 <th>change</th>
 <th>threshold low</th>
 <th>threshold high</th>
 <th>action</th>
 <th>create_user</th>
 </tr>
 ";
 
 $i=0;
 while ($i<$num) {
    $id        = mysql_result($result, $i, 'id');
    $create_dt = mysql_result($result, $i, 'create_dt');
    $create_user    = mysql_result($result, $i, 'create_user');
    $price     = mysql_result($result, $i, 'price');
    $low       = mysql_result($result, $i, 'low');
    $high      = mysql_result($result, $i, 'high');
    $volume    = mysql_result($result, $i, 'volume');
    $avg       = mysql_result($result, $i, 'avgexchange');
    $change    = mysql_result($result, $i, 'changepct');
    $myavg     = mysql_result($result, $i, 'myavg');
    $thlow     = mysql_result($result, $i, 'th_low');
    $thhigh    = mysql_result($result, $i, 'th_high');
    
    $deltaavg  = abs($avg - $myavg);
    
    $action = "-";
    
    if ($price<$thlow) {
        $action = "BUY";
        $bgcolor = "#6698FF";
    } else
    if ($price>$thhigh) {
        $action = "SELL";
        $bgcolor = "#E41B17";
    } else {
        $action = "-";
        $bgcolor = "#C0C0C0";
        if ($change<0) {
            $bgcolor = "#F5BCA9";
        } else 
        if ($change>0) {
          $bgcolor = "#BCF5A9";
        }
    }
    
    echo "
    <tr bgcolor='$bgcolor'>
    <td>$id</td>
    <td>$create_dt</td>
    <td>$price</td>
    <td>$low</td>
    <td>$high</td>
    <td>$volume</td>
    <td>$myavg</td>
    <td>$avg</td>
    <td>$deltaavg</td>
    <td>$change</td>
    <td>$thlow</td>
    <td>$thhigh</td>
    <td>$action</td>
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