<?php
/*
  This PHP script is used to upload workunits via HTTP upload
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/

 include("../utils/utils.inc.php");
 include("../utils/constants.inc.php");
 $debug=1;
 
 $wujob    = getparam("wujob", 0); 
 $wuresult = getparam("wuresult", 0); 
 
 if (($wujob==0) && ($wuresult==0)) die('ERROR: you need to set either wujob=1 or wuresult=1 as parameter');
 if (($wujob==1) && ($wuresult==1)) die('ERROR: you can not set both wujob=1 and wuresult=1 as parameter');
 

 if ($debug) echo "PHP script started\n";

 $type = $_FILES['myfile']['type'];
 if ($debug) echo "Type is $type.\n";

 if ($type == "application/octet-string") {
    $tmp_name = $_FILES['myfile']['tmp_name'];
    $size = $_FILES['myfile']['size']; 
	$filename = basename($_FILES['myfile']['name']);
    
	if ($wujob==1) {
	  $target_path = $workunitjobs_folder  . $filename;
	} else
	if ($wuresult==1) {
	  $target_path = $workunitresults_folder  . $filename;
	} else die('Internal error in php script');
	
	if ($debug) {
		echo "tmp_name is $tmp_name.\n";
		echo "size is $size.\n";
		echo "filename is $filename.\n";
		echo "target_path is $target_path.\n";
	}
	
	copy($tmp_name, $target_path);
  	if ($debug) echo "File uploaded succesfully\n";
	echo "OK\n";
 } else {
	echo "ERROR: type is incorrect, it should be 'application/octet-string' but was '$type' \n"; 
 }

 if ($debug) echo "PHP script over.\n";

?>