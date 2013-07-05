<html>
<head>
<title>GPU Server - Main page</title>
<link rel="stylesheet" type="text/css" href="stylesheet/gpu.css" />
</head>
<body>
<?php
include("utils/utils.inc.php");
include("utils/constants.inc.php");
include_once('utils/openflashchart/open_flash_chart_object.php');
include("conf/config.inc.php");

echo "<table>";
echo "<tr>";
include("head.inc.html");
echo "</tr>";
echo "<tr>";
include("menu.inc.html");

echo "<td>";
echo "<h2>Overview</h2>";
echo "<p>Welcome to $my_server_name $server_version by <a href='$my_homepage'>$my_username</a></p>";
echo "<table>";
echo "<tr><td>";
open_flash_chart_object( 500, 250, "http://" . $my_server_url . '/graph/graphdata-jobspermonth.php');
echo "</td><td>";
open_flash_chart_object( 500, 250, "http://" . $my_server_url . '/graph/graphdata-channels.php');
echo "</td></tr>";

echo "<tr><td>";
open_flash_chart_object( 500, 250, "http://" . $my_server_url . '/graph/graphdata-jobresultspermonth.php');
echo "</td><td>";
//open_flash_chart_object( 500, 250, "http://" . $my_server_url . '/graph/graphdata-scriptsperproject.php');
echo "</td>";

echo "</tr>";
echo "</table>";


?>
<?php include "bottom.inc.html" ?>
</td></tr>
</table>
</body>
</html>