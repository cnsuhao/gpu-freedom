<html>
<head>
<title>GPU Server - Main page</title>
</head>
<body>
<?php
include("utils/utils.inc.php");
include("utils/constants.inc.php");
include_once('utils/openflashchart/open_flash_chart_object.php');
include("conf/config.inc.php");
echo "<br>";
echo "<br>";
echo '<img src="images/gpu-inverse.jpg" border="0" />';
echo "<h1>Welcome to $my_server_name $server_version by <a href='$my_homepage'>$my_username</a></h1>";
echo "<hr>";
echo "<table>";
echo "<tr><td>";
open_flash_chart_object( 500, 250, "http://" . $my_server_url . '/graph/graphdata-jobspermonth.php');
echo "<br>";
echo "<center>";
echo "<b>Jobs<b><br>";
echo "<a href='jobqueue/list_jobdefinitions.php'>Definitions</a> <a href='jobqueue/list_jobqueues.php'>Queue</a> <a href='jobqueue/list_jobresults.php'>Results</a> <a href='jobqueue/list_jobstats.php'>Stats</a>";
echo "</center>";
echo "</td><td>";
open_flash_chart_object( 500, 250, "http://" . $my_server_url . '/graph/graphdata-channels.php');
echo "<center>";
echo "<b>Channels<b><br>";
echo "<a href='channel/list_channels.php'>Channels</a> <a href='channel/list_channel_messages.php'>Messages</a>";
echo "</center>";

echo "</td></tr>";

echo "<tr><td>";
open_flash_chart_object( 500, 250, "http://" . $my_server_url . '/graph/graphdata-jobresultspermonth.php');
echo "</td><td>";
echo "<center>";
echo "<b>Cluster<b><br>";
echo "<a href='cluster/list_clients.php'>Clients</a> <a href='supercluster/list_servers.php'>Servers</a>";
echo "</center>";
//open_flash_chart_object( 500, 250, "http://" . $my_server_url . '/graph/graphdata-scriptsperproject.php');
echo "</td></tr>";
echo "</table>";

?>
<hr>
<h6>GPU, a Global Processing Unit is Open Source under <a href="docs/GPL_license.txt">GPL</a> (source code <a href="http://sourceforge.net/projects/gpu/">here</a>) and is developed and mantained by the GPU Team</a>. 
</body>
</html>