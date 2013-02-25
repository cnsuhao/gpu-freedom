<?php
/*
  Constants used by GPU server
*/

// current server version
$server_version = 0.1;

// cluster settings
// maximum number of online nodes shown in list_clients.php
$max_online_clients_xml = 2000;

// maximum number of online nodes shown in list_servers.php
$max_online_servers_xml = 2000;

// this is the update interval in seconds of clients
// which touch the report_client.php script
// to report their online presence.
$client_update_interval = 60;

// this is the update interval in seconds of servers
$server_update_interval = 10; //3600;

// number of closest superservers which are informed of our server
$nb_superserver_informed = 2;
// timeout in seconds for a superserver to report the info
$max_superserver_timeout = 20;
?>