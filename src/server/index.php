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
echo "<h1>Welcome to $my_server_name $server_version</h1>";
echo "<hr>";
echo "<table>";
echo "<tr><td>";
open_flash_chart_object( 500, 250, "http://" . $my_server_url . '/graph/graphdata-jobspermonth.php');
echo "</td><td>";
//open_flash_chart_object( 500, 250, "http://" . $dns_name . '/graphdata-scriptspermodule.php');
echo "</td></tr>";

echo "<tr><td>";
//open_flash_chart_object( 500, 250, "http://" . $dns_name . '/graphdata-syncspermonth.php');
echo "</td><td>";
//open_flash_chart_object( 500, 250, "http://" . $dns_name . '/graphdata-scriptsperproject.php');
echo "</td></tr>";
echo "</table>";

?>
</body>
</html>