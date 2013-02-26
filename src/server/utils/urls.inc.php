<?php
/*
  Utilities to save and touch URLs
  
  Source code is under GPL, (c) 2002-2013 the Global Processing Unit Team
  
*/

function touch_url($url, $timeout) {
  $old = ini_set('default_socket_timeout', $timeout);
  $handle = fopen($url, 'r');
  ini_set('default_socket_timeout', $old);
  stream_set_timeout($handle, $timeout);
  stream_set_blocking($handle, 0); 
  fclose($handle);
}


function save_url($source_url, $target_file, $timeout) {
	$ch = curl_init();
	$fp = fopen ($target_file, 'w+');
	$ch = curl_init($source_url);
	curl_setopt($ch, CURLOPT_TIMEOUT, $timeout);
	curl_setopt($ch, CURLOPT_FILE, $fp);
	curl_setopt($ch, CURLOPT_FOLLOWLOCATION, 1);
	curl_setopt($ch, CURLOPT_ENCODING, "");
	curl_exec($ch);
	curl_close($ch);
	fclose($fp);
}


?>