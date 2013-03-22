<?php
/*
  Constants used by GPU server, do not change them! Typically, they are changed between releases, if necessary.
*/

// current server version
$server_version = 0.1;


// [Cluster Management]
// cluster settings
// maximum number of online nodes shown in list_clients.php
$max_online_clients_xml = 1000;

// maximum number of online nodes shown in list_servers.php
$max_online_servers_xml = 1000;

// this is the update interval in seconds of clients
// which touch the report_client.php script
// to report their online presence.
$client_update_interval = 60;

// this is the update interval in seconds of servers
$server_update_interval = 10; //3600;

// number of closest superservers which are informed of our server
$nb_superserver_informed = 2;
// timeout in seconds for a superserver to report our status or to retrieve servers or clients
$max_superserver_timeout = 20;


// [Job Management]
// this is the maximal number of requests allowed for a job definition
$max_requests_for_jobs = 500;

// How many jobs are distribued to a client, when it requests to crunch them
$jobs_to_be_distributed_at_once = 1;

// Retransmission interval in seconds for jobs which require an acknowledgment
$retransmission_interval = 7200;

// [Workunit Management]
$workunitjobs_folder    = "jobs/";
$workunitresults_folder = "results/";

?>