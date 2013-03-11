<?php
/*
  This script should be executed after the user filled some form and after logic creates the configuration file.
*/


 include("../conf/config.inc.php");	
 // defines parameters in the database
 include("../utils/parameters.inc.php");
 include("../utils/utils.inc.php");
 
  mysql_connect($dbserver, $username, $password);
 @mysql_select_db($database) or die("ERROR: Unable to select database, please check settings in conf/config.inc.php'\n");

 set_db_parameter("TEST", "DB_CONNECTION", "OK");
 
 set_db_parameter("CLIENT", "receive_servers_each", "3600");
 set_db_parameter("CLIENT", "receive_nodes_each", "120");
 set_db_parameter("CLIENT", "transmit_node_each", "180");
 set_db_parameter("CLIENT", "receive_jobs_each", "120");
 set_db_parameter("CLIENT", "transmit_jobs_each", "120");
 set_db_parameter("CLIENT", "receive_channels_each", "120");
 set_db_parameter("CLIENT", "transmit_channels_each", "120");
 set_db_parameter("CLIENT", "receive_chat_each", "45");
 set_db_parameter("CLIENT", "purge_server_after_failures", "30");
 
 
 // define an unique server id for this installation
 if (get_db_parameter("CONFIGURATION", "SERVER_ID", "missing")=="missing") {
	$serverid = create_unique_id();
    set_db_parameter("CONFIGURATION", "SERVER_ID", $serverid);
 }
 
 if (get_db_parameter("TIME", "UPTIME", "missing")=="missing") {
	set_db_parameter("TIME", "UPTIME", "0");
 }
 
 if (get_db_parameter("TIME", "LAST_SUPERSERVER_CALL", "missing")=="missing") {
	set_db_parameter("TIME", "LAST_SUPERSERVER_CALL", time());
 }
 

 if (get_db_parameter("SECURITY", "PWD_HASH_SALT", "missing")=="missing") {
	// TODO: will this be just the next random number after serverid?
	$salt     = create_unique_id();
	set_db_parameter("SECURITY", "PWD_HASH_SALT", $salt);
 }
  
 mysql_close();
 
 echo "OK";
?>