<?php
// mySQL settings
$username="gpuuser";
$password="";
$database="gpu_server";
$dbserver="localhost";

// cluster settings
// maximum number of online nodes shown in list_clients_online_xml.php
$max_online_clients_xml = 200;

// this is the update interval in seconds of clients
// which touch the report_client.php script
// to report their online presence.
$update_interval = 300;

// in list pages, how many entries are displayed at once
$entriesperpage = 25;

?>
