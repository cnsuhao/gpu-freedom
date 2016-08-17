<html>
<head>
<title>Store News CNN</title>
</head>
<body>
<?php
 include("../lib/spiderbook/LIB_parse.php");
 include_once('../../../server/utils/mydql2i/mysql2i.class.php');
 include("conf/config.inc.php");
 
 echo "<h3>Store News CNN</h3>";
 echo "<p>Parsing...</p>";
 $frequency=-1;
 $networkdiff=-1;
 
 mysql_connect($dbserver, $username, $password);
@mysql_select_db($database) or die("Unable to select database");

 $lines = file("cnn.html");
 $hugepage = "";
 
 for ( $i = 0; $i < sizeof( $lines ); $i++ ) {
    $hugepage = $hugepage . $lines[$i] . "\n";
 }
 
 
 $news= parse_array($hugepage, '<span class="cd__headline-text">', '</span>');
 
 for ( $i = 0; $i < sizeof( $news ); $i++ ) {

    $mynews = addslashes($news[$i]); 
	if ( 
	      ($mynews!="") && 
		  //(strpos($mynews, "notify-box") == false) && 
		  strlen($mynews)>10
		)  
		{
		
		 $query_check="select count(*) from tbnews where newstitle='$mynews' and (create_dt>=NOW() - INTERVAL 2 DAY) and source='CNN'";
	     $res_check = mysql_query($query_check);
		 if ($res_check!="") $count=mysql_result($res_check, 0, "count(*)"); else $count=0;
		 
		 echo "<p>$mynews</p>";
		 echo "$count\n";
		 
		 
		 if ($count==0) {
			$query="INSERT INTO tbnews (create_dt, newstitle, source) 
					VALUES(NOW(), '$mynews', 'CNN');";
			if (!mysql_query($query)) echo mysql_error();
		 
		 } 
		 
	}
 }
 
 mysql_close();
 
?>
</body>
</html>
