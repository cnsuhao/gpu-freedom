<?php
/*
  This PHP script is used to test the HTTP upload capability of the server
  
  Source code is under GPL, (c) 2002-2011 the Global Processing Unit Team
  
*/
 include("../conf/config.inc.php");
 $type = $_FILES['myfile']['type'];
 if ($type == "application/octet-string") {
    $tmp_name = $_FILES['myfile']['tmp_name'];
    $size = $_FILES['myfile']['size']; 
	$filename = basename($_FILES['myfile']['name']);
	
	//TODO: check if $rootwu configuration path is needed
	$target_path = "$rootwu/$filename";
	copy($tmp_name, $target_path);	
 }
?>