<?php

function phone_home() {
  // a connection needs to be established
  include("utils/constants.inc.php");
  // collect data for the phone call
  $nbscripts  = count_records("tbscript");
  $nbmodules  = count_records("tbmodule");
  $nbprojects = count_records("tbproject");
  $nbbranches = count_records("tbbranch");
  $nbsyncs    = count_records("tbsyncstats");
  $nbusers    = count_records("tbuser");
  $nbmp       = count_records("tbmoduleproject");
  $nbsb       = count_records("tbscriptbranch");
  
  $timeout = 6;
  $old = ini_set('default_socket_timeout', $timeout);
  $handle = fopen("http://deltasql.sourceforge.net/deltasql/phone_call.php?nbscripts=$nbscripts&nbmodules=$nbmodules&nbprojects=$nbprojects&nbbranches=$nbbranches&nbsyncs=$nbsyncs&nbusers=$nbusers&nbmp=$nbmp&nbsb=$nbsb&version=$deltasql_version", 'r');
  ini_set('default_socket_timeout', $old);
  stream_set_timeout($handle, $timeout);
  stream_set_blocking($handle, 0); 

  fclose($handle);

  //die("<b>$nbscripts $nbmodules $nbprojects $nbbranches $nbsyncs $nbusers $nbmp $nbsb $deltasql_version</b>");
}

function answer_phone($ip, $nbscripts, $nbmodules, $nbprojects, $nbbranches, $nbsyncs, $nbusers, $nbmp, $nbsb, $deltasql_version) {
  // a connection needs to be established
   $query="INSERT INTO tbstats (id, ip, deltasql_version, create_dt, nbscripts, nbmodules, nbprojects, nbbranches, nbsyncs, nbusers, nbmp, nbsb ) VALUES ('', '$ip', '$deltasql_version', NOW(), $nbscripts, $nbmodules, $nbprojects, $nbbranches, $nbsyncs, $nbusers, $nbmp, $nbsb);"; 
   mysql_query($query);
}



?>