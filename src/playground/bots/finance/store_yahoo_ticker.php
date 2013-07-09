<html>
<head>
<title>Store Yahoo Ticker</title>
</head>
<body>
<?php
 include("spiderbook/LIB_parse.php");
 include("conf/config.inc.php");
 if (isset($_GET['ticker'])) $ticker = $_GET['ticker']; else $ticker="";
 
 if ($ticker=="") {
	die("<b>Ticker parameter not defined!</b>");
 }
  
 echo "<h3>Storing Yahoo Ticker $ticker</h3>";
 echo "<p>Parsing...</p>";
 
 // we are intersted in:
 // <span class="time_rtq_ticker"><span id="yfs_l10_^vix">14.78</span></span>
 
 $lines = file("finance.html");
 $hugepage = '';
 for ( $i = 0; $i < sizeof( $lines ); $i++ ) {
   $hugepage = $hugepage . $lines[$i];
 }
 
 $table = return_between($hugepage, "<span class=\"time_rtq_ticker\">", '</span>', INCL);
 echo "<br><b>Table</b><br>";
 echo $table;
 
 echo "<br><b>Value</b><br>";
 $value = trim(return_between($hugepage, "<span id=\"yfs_l10_^$ticker\">", '</span>', EXCL));
 $myvalue = str_replace(',','',$value);
 echo "$myvalue<br>";
 
 
 echo "<p>Connecting...</p>";
 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");
 $query="INSERT INTO tickers (create_dt, name, value, changepct, create_user) 
         VALUES(NOW(), '$ticker', $myvalue, 0, 'php');";
 
 echo "<p>*$query*</p>";                                 
 if (!mysql_query($query)) echo mysql_error();
 mysql_close();
 echo "<p>Over.</p>";
?>
</body>
</html>