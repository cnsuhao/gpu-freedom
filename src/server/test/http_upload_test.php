<?php
/*
  This PHP script is used to test the HTTP upload capability of the server
  
  Source code is under GPL, (c) 2002-2011 the Global Processing Unit Team
  
*/
 // echo "PHP script started\n";
 include("../conf/config.inc.php");
 $type = $_FILES['myfile']['type'];
 // echo "Type is $type.\n";

 if ($type == "application/octet-string") {
    $tmp_name = $_FILES['myfile']['tmp_name'];
    $size = $_FILES['myfile']['size']; 
	$filename = basename($_FILES['myfile']['name']);
    /*
	echo "tmp_name is $tmp_name.\n";
	echo "size is $size.\n";
	echo "filename is $filename.\n";
	*/
	
	$target_path = "$filename";
	copy($tmp_name, $target_path);
  	//echo "File uploaded succesfully\n";
	echo "OK\n";
 } else {
	echo "FAIL\n"; 
 }
 //echo "PHP script over.\n";

?>