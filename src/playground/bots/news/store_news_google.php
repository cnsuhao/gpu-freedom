<html>
<head>
<title>Store News Google</title>
</head>
<body>
<?php
 include("../lib/spiderbook/LIB_parse.php");
 include_once('../../../server/utils/mydql2i/mysql2i.class.php');
 include("conf/config.inc.php");
 
 echo "<h3>Store News Google</h3>";
 echo "<p>Parsing...</p>";
 $frequency=-1;
 $networkdiff=-1;
 
 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");

 $lines = file("news.html");
 for ( $i = 0; $i < sizeof( $lines ); $i++ ) {
	$news = return_between($lines[$i], '<span class="titletext">', '</span>', EXCL);
    if ( 
	      ($news!="") && 
		  (strpos($news, "notify-box") == false) && 
		  (strpos($news, "2016 Google")==false) &&
		  (strpos($news, "display: none")==false) &&
		  (strpos($news, "2016 Google")==false) &&
		  (strpos($news, "www.google.com")==false) &&
		  strlen($news)>10
		)  
		{
		
		 $query_check="select count(*) from tbnews where newstitle='$news' and (create_dt>=NOW() - INTERVAL 1 DAY)";
	     $res_check = mysql_query($query_check);
		 $count=mysql_result($res_check, 0, "count(*)");
		 
		 echo "<p>$news</p>";
		 echo "$count\n";
		 
		 if ($count==0) {
			$query="INSERT INTO tbnews (create_dt, newstitle, source) 
					VALUES(NOW(), '$news', 'GOOGLENEWS');";
			if (!mysql_query($query)) echo mysql_error();
		 
		 } 
		 
	}
 }
 
 mysql_close();
 
?>
</body>
</html>