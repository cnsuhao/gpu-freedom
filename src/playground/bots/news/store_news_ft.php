<html>
<head>
<title>Store News Financial Times</title>
</head>
<body>
<?php
 include("../lib/spiderbook/LIB_parse.php");
 include_once('../../../server/utils/mydql2i/mysql2i.class.php');
 include("conf/config.inc.php");
 
 echo "<h3>Store News Financial Times</h3>";
 echo "<p>Parsing...</p>";

 
 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");

 $lines = file("ft.html");
 for ( $i = 0; $i < sizeof( $lines ); $i++ ) {
	$news = return_between($lines[$i], '<h4>', '</h4>', EXCL);
	$news = return_between($news."\/", '">', '\/', EXCL);
	if ( 
	      ($news!="") && 
		  (strpos($news, "class=") == false) && 
		  (strpos($news, "span>TRAVEL</span>") == false) && 		  
		  (strpos($news, "span>HOUSE &amp;") == false) && 		  
		  strlen($news)>12
		)  
		{
	     $news = addslashes($news);
    	
		 $query_check="select count(*) from tbnews where newstitle='$news' and (create_dt>=NOW() - INTERVAL 7 DAY) and source='FINANCIALTIMES'";
	     $res_check = mysql_query($query_check);
		 if ($res_check!="") $count=mysql_result($res_check, 0, "count(*)"); else $count=0;
		 
		 echo "<p>*$news*</p>";
		 echo "$count\n";
		 
		 if ($count==0) {
			$query="INSERT INTO tbnews (create_dt, newstitle, source) 
					VALUES(NOW(), '$news', 'FINANCIALTIMES');";
			if (!mysql_query($query)) echo mysql_error();
		 
		 } 
		 
	}
 }
 
 mysql_close();
 
?>
</body>
</html>