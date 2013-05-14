<html>
<head>
<title>Store Bitcoin Value</title>
</head>
<body>
<?php
 include("head.inc.php");
 include("spiderbook/LIB_parse.php");
 include("conf/config.inc.php");
 include("utils/constants.inc.php");

 echo "<h3>Store Bitcoin Value</h3>";
 echo "<p>Parsing...</p>";

 $lines = file("bitcoin.html");
 $lastprice=-1;
 $high=-1;
 $low=-1;
 $volume=-1;
 $avg=-1;
 
 echo "<table>";
 for ( $i = 0; $i < sizeof( $lines ); $i++ ) {
	if (strpos($lines[$i], '<li id="lastPrice">')>0) {
         $lastprice=return_between($lines[$i], '<li id="lastPrice">Last price:<span>$', '</span></li>', EXCL);
         echo "<tr><td>Last Price:</td><td>$lastprice</td></tr>";
    } 
    
    if (strpos($lines[$i], '<li id="highPrice">')>0) {
         $high=return_between($lines[$i], '<li id="highPrice">High:<span>$', '</span></li>', EXCL);
         echo "<tr><td>High:</td><td>$high</td></tr>";
    } 
    
    if (strpos($lines[$i], '<li id="lowPrice">')>0) {
         $low=return_between($lines[$i], '<li id="lowPrice">Low:<span>$', '</span></li>', EXCL);
         echo "<tr><td>Low:</td><td>$low</td></tr>";
    } 
    
    if (strpos($lines[$i], '<li id="volume">')>0) {
         $volume=return_between($lines[$i], '<li id="volume">Volume:<span>', ' BTC</span></li>', EXCL);
         echo "<tr><td>Volume:</td><td>$volume</td></tr>";
    }
    
    if (strpos($lines[$i], '<li id="weightedAverage">')>0) {
         $avg=return_between($lines[$i], '<li id="weightedAverage">Weighted Avg:<span>$', '</span></li>', EXCL);
         echo "<tr><td>Weighted Average:</td><td>$avg</td></tr>";
    } 
    
    
 } 
 echo "</table>";

 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");
 
 echo "<p>Calculating thresholds...</p>";
 
 $thlowquery = "select avg(low) from pricevalue where create_dt >= (NOW() - INTERVAL $th_day_interval DAY);";
 echo "<p>$thlowquery</p>";
 $thlowresult = mysql_query($thlowquery);
 $th_low=mysql_result($thlowresult, 0, "avg(low)");
 
 $thhighquery = "select avg(high) from pricevalue where create_dt >= (NOW() - INTERVAL $th_day_interval DAY);";
 echo "<p>$thhighquery</p>";
 $thhighresult = mysql_query($thhighquery);
 $th_high=mysql_result($thhighresult, 0, "avg(high)");
 
 $avgquery = "select avg(avgexchange) from pricevalue where create_dt >= (NOW() - INTERVAL $th_day_interval DAY);";
 echo "<p>$avgquery</p>";
 $avgresult = mysql_query($avgquery);
 $myavg=mysql_result($avgresult, 0, "avg(avgexchange)");
  
 echo "<p>Storing...</p>";
 
 if ($lastprice!=-1) {
    $query="INSERT INTO pricevalue (id, create_dt, price, high, low, volume, avgexchange, myavg, th_low, th_high) VALUES('',
                                    NOW(), $lastprice, $high, $low, $volume, $avg, $myavg, $th_low, $th_high);";
                                 
 } else {
    $query="INSERT INTO log (id, message, level, create_dt) VALUES('', 'Could not retrieve prices!', 'ERROR', NOW());";
 }
 echo "<p>$query</p>";                                 
 mysql_query($query);
 mysql_close();
 
 echo "<p>Over.</p>";
?>
</body>
</html>