<?php

/*
 We use the Free IP GeoLocation/GeoIp API provided by
 http://geoip.nekudo.com/
*/


function get_geoip_info($ip, $full_info=false) {
  $curl = curl_init();

  $geoip_server = "http://geoip.nekudo.com/api/";
  $proxy_for_geoip = '192.168.4.2:8080';
  
  curl_setopt($curl, CURLOPT_HTTPHEADER, array('Content-Type: application/json')); 
  if ($full_info) 
		$geoip_url = $geoip_server . $ip . "/full";  
  else		
		$geoip_url = $geoip_server . $ip;  
  

  curl_setopt($curl, CURLOPT_URL, $geoip_url);  
  curl_setopt($curl, CURLOPT_RETURNTRANSFER, true);
  curl_setopt($curl, CURLOPT_PROXY, $proxy_for_geoip);
  
  $result = curl_exec($curl);
  $resarray = json_decode($result, true);
  
  
  return $resarray; 
}
?>