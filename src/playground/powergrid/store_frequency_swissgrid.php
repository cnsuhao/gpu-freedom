<html>
<head>
<title>Store Frequency Swissgrid</title>
</head>
<body>
<?php
 include("spiderbook/LIB_parse.php");
 include("conf/config.inc.php");
 
 echo "<h3>Store Frequency Swissgrid</h3>";
 echo "<p>Parsing...</p>";
 $frequency=-1;
 $networkdiff=-1;
 
 $lines = file("frequency.html");
 $hugepage = '';
 for ( $i = 0; $i < sizeof( $lines ); $i++ ) {
   if (strpos($lines[$i], 'frequencymark.gif"')>0) $lines[$i]='    src="gadgets/netfrequency/img/frequencymark.gif"';
   if (strpos($lines[$i], 'frequencyband.gif"')>0) $lines[$i]='    src="gadgets/netfrequency/img/frequencyband.gif"';
   
   $hugepage = $hugepage . $lines[$i];
 }
 $table = return_between($hugepage, '<table class="data">', '</table>', INCL);
 echo "<br><b>Table</b><br>";
 echo $table;
 
 $fh = fopen("frequencygraph.html", 'w');
 fwrite($fh, "$table");
 fclose($fh);
 
 echo "<br><b>Strings</b><br>";
 $frequencyrow = return_between($table, '<strong>Aktuelle Frequenz</strong>','</tr>', EXCL);
 $frequencystr = return_between($frequencyrow, '<span>','</span>', EXCL);
 echo "*$frequencystr*<br>";
 
 $netdiffrow = return_between($table, '<strong>Aktuelle Netzabweichung</strong>','</tr>', EXCL);
 $netdiffstr = return_between($netdiffrow, '<span>','</span>', EXCL);
 echo "*$netdiffstr*<br>";
 
 echo "<br><b>Values</b><br>";
 $frequency = str_replace(" Hz", "", $frequencystr);
 $frequency = str_replace(",", ".", $frequency);
 echo "*$frequency*<br>";
 
 $netdiff = str_replace(" s", "", $netdiffstr);
 $netdiff = str_replace(",", ".", $netdiff);
 echo "*$netdiff*<br>";

 echo "<p>Connecting...</p>";
 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");
 
 $query="INSERT INTO frequency (id, frequency, networkdiff, controlarea, tso, create_dt, create_user) 
                                VALUES('', $frequency, $netdiff, 'SG_ST', 'Swissgrid', NOW(), 'php');";
 
 echo "<p>$query</p>";                                 
 mysql_query($query);
 
 mysql_close();
 echo "<p>Over.</p>";
 
?>
</body>
</html>