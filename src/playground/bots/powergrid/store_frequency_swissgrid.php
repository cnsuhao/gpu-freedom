<html>
<head>
<title>Store Frequency Swissgrid</title>
</head>
<body>
<?php
 include("../lib/spiderbook/LIB_parse.php");
 include("conf/config.inc.php");
 
 echo "<h3>Store Frequency Swissgrid</h3>";
 echo "<p>Parsing...</p>";
 $frequency=-1;
 $networkdiff=-1;
 
 $lines = file("frequency.html");
 $hugepage = '';
 $flag = 1;
 for ( $i = 0; $i < sizeof( $lines ); $i++ ) {
   if (strpos($lines[$i], 'frequencymark.gif"')>0) $lines[$i]='   <img style="margin-left: 0px; margin-right: 0px" src="gadgets/netfrequency/img/frequencymark.gif" alt="|" />';
   if (strpos($lines[$i], 'frequencyband.gif"')>0) $lines[$i]='   <img style="margin-left: 0px; margin-right: 0px" src="gadgets/netfrequency/img/frequencyband.gif" alt="Frequenzspektrum" /></div>';
   
   // ugly, but parse_array does not work for some strange reason
   if (strpos($lines[$i], '<span>')>0) {
            if ($flag==1) {
                $frequencystr=$lines[$i];
                $flag=0;
            } else {
                $netdiffstr=$lines[$i];
            }
   }
   $hugepage = $hugepage . $lines[$i];
 }
 $table = return_between($hugepage, '<table class="data">', '</table>', INCL);
 echo "<br><b>Table</b><br>";
 echo $table;
 
 $fh = fopen("frequencygraph.html", 'w');
 fwrite($fh, "$table");
 fclose($fh);
 
 echo "<br><b>Strings</b><br>";
 
 $frequencystr = return_between($frequencystr, '<span>', '</span>', EXCL);;
 echo "*$frequencystr*<br>";
 
 $netdiffstr = return_between($netdiffstr, '<span>', '</span>', EXCL);;
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
 $query="INSERT INTO tbfrequency (create_dt, controlarea, tso, create_user, frequencyhz, networkdiff) 
         VALUES(NOW(), 'SG_ST', 'Swissgrid', 'script', $frequency, $netdiff);";
 
 echo "<p>*$query*</p>";                                 
 if (!mysql_query($query)) echo mysql_error();
 mysql_close();
 echo "<p>Over.</p>";
 
?>
</body>
</html>