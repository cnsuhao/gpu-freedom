<html>
<head>
<title>Store News Arstechnica</title>
</head>
<body>
<?php
 include("../lib/spiderbook/LIB_parse.php");
 include_once('../../../server/utils/mydql2i/mysql2i.class.php');
 include("conf/config.inc.php");
 
 echo "<h3>Store News Arstechnica</h3>";
 echo "<p>Parsing...</p>";
 $frequency=-1;
 $networkdiff=-1;
 
 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");

 $lines = file("arstechnica.html");
 for ( $i = 0; $i < sizeof( $lines ); $i++ ) {
	$news = return_between($lines[$i], '<h2><a href="', '</a></h2>', EXCL);
	$news = return_between($news."\/", '/">', '\/', EXCL);
	$news = addslashes($news);
    if ( 
	      ($news!="") && 
		  //(strpos($news, "notify-box") == false) && 
		  strlen($news)>10
		)  
		{
		
		 $query_check="select count(*) from tbnews where newstitle='$news' and (create_dt>=NOW() - INTERVAL 14 DAY) and source='ARSTECHNICA'";
	     $res_check = mysql_query($query_check);
		 if ($res_check!="") $count=mysql_result($res_check, 0, "count(*)"); else $count=0;
		 
		 echo "<p>$news</p>";
		 echo "$count\n";
		 
		 if ($count==0) {
			$query="INSERT INTO tbnews (create_dt, newstitle, source) 
					VALUES(NOW(), '$news', 'ARSTECHNICA');";
			if (!mysql_query($query)) echo mysql_error();
		 
		 } 
		 
	}
 }
 
 mysql_close();
 
?>
</body>
</html>
