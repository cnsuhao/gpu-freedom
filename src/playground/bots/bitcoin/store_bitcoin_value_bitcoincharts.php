<html>
<head>
<title>Store Bitcoin Value</title>
</head>
<body>
<?php
 include("head.inc.php");
 include("../lib/spiderbook/LIB_parse.php");
 include("conf/config.inc.php");
 include("utils/constants.inc.php");

 echo "<h3>Store Bitcoin Value</h3>";
 echo "<p>Parsing...</p>";

 $lastprice=-1;
 $high=-1;
 $low=-1;
 $volume=-1;
 $avg=-1;
 
 $lines = file("bitcoincharts.html");
 $hugepage = '';
 for ( $i = 0; $i < sizeof( $lines ); $i++ ) {
   $hugepage = $hugepage . $lines[$i];
 }
 $table = return_between($hugepage, '<table class="data">', '</table>', INCL);
 echo $table;
 
 $mtgox = return_between($table, '<a href="/markets/mtgoxUSD.html">mtgoxUSD</a>','</tr>', EXCL); 
 $mtgox = str_replace ('<td class="right">', '' , $mtgox);
 $mtgox = str_replace ('</td>', '' , $mtgox);
 
 echo "<pre>";
 echo $mtgox;
 echo "</pre>";
 
 $lines = explode("\n", $mtgox);
 echo "<pre>";
 for ( $i = 0; $i < sizeof( $lines ); $i++ ) {
    echo $i . ' '. $lines[$i] . ' ';
 }
 echo "</pre>";
 
 
 $lastprice = trim($lines[2]);
 $avg       = trim($lines[3]);
 $change    = trim($lines[4]);
 $volume    = trim($lines[5]);
  
 $change = str_replace ('%', '' , $change); 
 $change = str_replace ('+', '' , $change); 
 $change = $change / 100;
 
 $volume = $volume / 30;
 
 echo "<table>";
         echo "<tr><td>Last Price:</td><td>$lastprice</td></tr>";
         echo "<tr><td>Exchange Average:</td><td>$avg</td></tr>";
         echo "<tr><td>Change:</td><td>$change</td></tr>";
         echo "<tr><td>Volume:</td><td>$volume</td></tr>";
 echo "</table>";

 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");
 
 echo "<p>Calculating thresholds...</p>";
 
 $thlowquery = "select min(price) from pricevalue where create_dt >= (NOW() - INTERVAL $th_day_interval DAY);";
 echo "<p>$thlowquery</p>";
 $thlowresult = mysql_query($thlowquery);
 $th_low=mysql_result($thlowresult, 0, "min(price)");
 
 $thhighquery = "select max(price) from pricevalue where create_dt >= (NOW() - INTERVAL $th_day_interval DAY);";
 echo "<p>$thhighquery</p>";
 $thhighresult = mysql_query($thhighquery);
 $th_high=mysql_result($thhighresult, 0, "max(price)");
 
 
 $highquery = "select max(price) from pricevalue where create_dt >= (NOW() - INTERVAL 5 DAY);";
 echo "<p>$highquery</p>";
 $highresult = mysql_query($highquery);
 $high=mysql_result($highresult, 0, "max(price)");

 $lowquery = "select min(price) from pricevalue where create_dt >= (NOW() - INTERVAL 5 DAY);";
 echo "<p>$lowquery</p>";
 $lowresult = mysql_query($lowquery);
 $low=mysql_result($lowresult, 0, "min(price)");
 
 
 $avgquery = "select avg(price) from pricevalue where create_dt >= (NOW() - INTERVAL $th_day_interval DAY);";
 echo "<p>$avgquery</p>";
 $avgresult = mysql_query($avgquery);
 $myavg=mysql_result($avgresult, 0, "avg(price)");
  
 echo "<p>Storing...</p>";
 
 if ($lastprice!="") {
    $query="INSERT INTO pricevalue (id, create_dt, price, high, low, volume, avgexchange, myavg, th_low, th_high, changepct, create_user) VALUES('',
                                    NOW(), $lastprice, $high, $low, $volume, $avg, $myavg, $th_low, $th_high, $change, 'mainloop');";
                                 
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