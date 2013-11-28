<?php 
session_start(); 
include("../session/login.inc.php");
if (!isset($_SESSION['userbot'])) $_SESSION['userbot']="";
if (!isset($_SESSION['userpwd'])) $_SESSION['userpwd']="";
if (($_SESSION['userbot'] <> $username) || ($_SESSION['userpwd'] <> $password)) 
   die("<html><body><a href='../session/login.php'>Please login</a></body></html>");
?>
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

 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");


 function printWallet($name) {
		$query="select * from wallet where (name='$name') and (id=(select max(w.id) from wallet w where w.name='$name'))";
        $result = mysql_query($query);
        
		$id          = mysql_result($result, 0, 'id');
		$btc         = mysql_result($result, 0, 'btc');
		$usd         = mysql_result($result, 0, 'usd');
		$price       = mysql_result($result, 0, 'marketprice_usd');
		$total       = mysql_result($result, 0, 'total_usd');
		$bucket      = mysql_result($result, 0, 'bucket_usd');
		$createdt       = mysql_result($result, 0, 'create_dt');
		
		echo "
		<tr BGCOLOR=\"#FDD017\">
		<td>$id</td>
		<td>$name</td>
		<td>$btc</td>
		<td>$usd</td>
		<td>$price</td>
		<td>$total</td>
		<td>$bucket</td>
		<td>$createdt</td>
		</tr>
		";
 }

 
 echo "<p><tt>./mainloop.sh to update prices</tt></p>";
 
 echo "<table>";
 echo "<tr><td>";
 open_flash_chart_object( 500, 250, $dns_name . '/graph/lastprices.php');
 echo "</td>";

 echo "<td>";
 /* last trades done by bot */
 $query="select * from trade order by id desc LIMIT 10;";
 $result = mysql_query($query);
 $num = mysql_numrows($result);
 
 echo "
 <table border='1'>
 <tr>
 <th>id</th>
 <th>direction</th>
 <th>amount</th>
 <th>price</th>
 <th>total</th>
 <th>marketorder</th>
 <th>description</th>
 <th>create_dt</th>
 </tr>
 ";
 
 $i=0;
 while ($i<$num) {
    $id          = mysql_result($result, $i, 'id');
    $direction  = mysql_result($result, $i, 'direction');
    $amount    = mysql_result($result, $i, 'amount');
    $price     = mysql_result($result, $i, 'price');
    $total       = mysql_result($result, $i, 'total');
    $mo          = mysql_result($result, $i, 'marketorder');
    $desc       = mysql_result($result, $i, 'description');
    $createdt       = mysql_result($result, $i, 'create_dt');
	
	if ($direction=="BUY") {
        $bgcolor = "#6698FF";
    } else {
        $bgcolor = "#E41B17";
    }
 
    echo "
	<tr bgcolor='$bgcolor'>
	<td>$id</td>
	<td>$direction</td>
	<td>$amount</td>
	<td>$price</td>
	<td>$total</td>
	<td>$mo</td>
	<td>$desc</td>
	<td>$createdt</td>
	</tr>
	";
	
	$i++;
 }
 
 echo "</table></td>";

 echo "<td>";
 open_flash_chart_object( 500, 250, $dns_name . '/graph/lastvolumes.php');
 echo "</td>";
  
 echo '<td><img src="pictures/linuxbc.jpg" border=0 /></td>';
 echo "</tr>";
 echo "</table>";
 
 echo "<hr>";

 echo "<table>"; 
 echo "<td>";
 $query="select * from pricevalue order by id desc LIMIT 10;";
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
 <th></th>
 <th>bid</th>
 <th>ask</th>
 <th>spread</th>
 <th></th>
 <th>vwap</th>
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
    $buy       = mysql_result($result, $i, 'buy');
    $sell      = mysql_result($result, $i, 'sell');
    $vwap       = mysql_result($result, $i, 'buy');
	$spread    = $sell-$buy;
    
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
	<td></td>
    <td>$buy</td>
    <td>$sell</td>
    <td>$spread</td>
	<td></td>
	<td>$vwap</td>
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
 
 echo "</td>";
 
 echo "<td>";
 /* wallets*/
 echo "<table border='1'>";
 echo "<tr>
      <th>id</th>
	  <th>name</th>
	  <th>btc</th>
	  <th>usd</th>
	  <th>marketprice</th>
	  <th>total</th>
	  <th>bucket</th>
	  <th>create_dt</th>
      </tr>	  
      ";
 printWallet("shortterm");
 printWallet("midterm");
 printWallet("longterm");
 printWallet("tiz");
 printWallet("total");
 printWallet("mtgox");
 echo "</table>";
 
 /*
 //from http://www.1archive.com/java/lunarphases/
 echo "<br>";
 echo '<center><applet width="400" height="168" align="middle" code="LunarPhases.class" CODEBASE="LunarPhases/"></center>';
 */
 
 echo "</td>";
 echo "</table>"; 
 
 
 mysql_close();

 
 
 ?>

</body>
</html>