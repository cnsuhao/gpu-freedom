<?php

/*
 We use the Free IP GeoLocation/GeoIp API provided by
 http://geoip.nekudo.com/
*/

  $curl = curl_init();
  curl_setopt($curl, CURLOPT_HTTPHEADER, array('Content-Type: application/json')); 
  curl_setopt($curl, CURLOPT_URL, 'http://geoip.nekudo.com/api/212.243.72.167');  
  curl_setopt($curl, CURLOPT_RETURNTRANSFER, true);
  curl_setopt($curl, CURLOPT_PROXY, '192.168.4.2:8080');
  
  $result = curl_exec($curl);
  
  $result = json_decode($result, true);

  var_dump($result)
  
?>