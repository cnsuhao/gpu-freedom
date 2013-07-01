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
   if (strpos($lines[$i], 'frequencymark.gif"')>0) $lines[$i]='   <img style="margin-left: 0px; margin-right: 0px" src="gadgets/netfrequency/img/frequencymark.gif" alt="|" />';
   if (strpos($lines[$i], 'frequencyband.gif"')>0) $lines[$i]='   <img style="margin-left: 0px; margin-right: 0px" src="gadgets/netfrequency/img/frequencyband.gif" alt="Frequenzspektrum" /></div>';
   
   $hugepage = $hugepage . $lines[$i];
 }
 $table = return_between($hugepage, '<table class="data">', '</table>', INCL);
 echo "<br><b>Table</b><br>";
 echo $table;
 
 $fh = fopen("frequencygraph.html", 'w');
 fwrite($fh, "$table");
 fclose($fh);
 
 echo "<br><b>Strings</b><br>";
 $spans=parse_array($table,"<span>","</span>",EXCL);
 
 $frequencystr = $spans[0];
 echo "*$frequencystr*<br>";
 
 $netdiffstr = $spans[1];
 echo "*$netdiffstr*<br>";

 echo "<br><b>Values</b><br>";
 $frequency = str_replace(" Hz", "", $frequencystr);
 //$frequency = str_replace(",", ".", $frequency);
 echo "*$frequency*<br>";
 
 $netdiff = str_replace(" s", "", $netdiffstr);
 //$netdiff = str_replace(",", ".", $netdiff);
 echo "*$netdiff*<br>";

 echo "<p>Connecting...</p>";
 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");
 
 $query="INSERT INTO frequency (id, frequencyhz, networkdiff, controlarea, tso, create_dt, create_user) VALUES('', ".$frequency.", ".$netdiff.", 'SG_ST', 'Swissgrid', NOW(), 'php');";
 
 echo "<p>$query</p>";                                 
 $result=mysql_query($query);
 echo "Result: *$result*"; 
 mysql_close();
 echo "<p>Over.</p>";

 
?>
</body>
</html>