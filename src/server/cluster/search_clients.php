<html>
<title>
GPU Server - Search Among Clients
</title>
<body>
<img src="../images/gpu-inverse.jpg" border="0"><br>

<?php
include("../conf/config.inc.php");
include("../utils/utils.inc.php");

$cookie_nodename=$_COOKIE['search_nodename'];
$cookie_cputype=$_COOKIE['search_cputype'];
$cookie_os=$_COOKIE['search_os'];
$cookie_country=$_COOKIE['search_country'];
$cookie_mhzfrom=$_COOKIE['search_mhzfrom'];
$cookie_mhzto=$_COOKIE['search_mhzto'];
$cookie_ramfrom=$_COOKIE['search_ramfrom'];
$cookie_ramto=$_COOKIE['search_ramto'];
$cookie_uptimefrom=$_COOKIE['search_uptimefrom'];
$cookie_uptimeto=$_COOKIE['search_uptimeto'];
$cookie_totuptimefrom=$_COOKIE['search_totuptimefrom'];
$cookie_totuptimeto=$_COOKIE['search_totuptimeto'];
$cookie_fromversion=$_COOKIE['search_fromversion'];
$cookie_toversion=$_COOKIE['search_toversion'];
?>

<h3>Search Among Computers</h3>
<form name="searchscripts" id="searchscripts" method="post" action="search_clients_store_cookies.php">
<table>
<tr>
<td><b>Client name:</b></td> <td><input type="text" name="nodename" 
                                           value="<?php echo "$cookie_nodename"; ?>" size="40">

</td>
</tr>
<tr>
<td><b>CPU Type:</b></td> <td><input type="text" name="cputype" 
                                           value="<?php echo "$cookie_cputype"; ?>" size="40">
</td>
</tr>
<tr>
<td><b>Operating System:</b></td> <td><input type="text" name="os" 
                                            value="<?php echo "$cookie_os"; ?>" size="40">
 </td>
</tr>
<tr>
<td><b>Country:</b></td> <td><input type="text" name="country" 
                                            value="<?php echo "$cookie_country"; ?>" size="40">
 </td>
</tr>
<tr>
<td><b>MegaHertz</b></td> 
<td>From: <input type="text" name="mhzfrom" value="<?php echo "$cookie_mhzfrom"; ?>" size="8">
 To: <input type="text" name="mhzto" value="<?php echo "$cookie_mhzto"; ?>" size="8"></td>
</td>
</tr>
<tr>
<td><b>RAM (in MegaByte)</b></td> 
<td>From: <input type="text" name="ramfrom" value="<?php echo "$cookie_ramfrom"; ?>" size="8">
 To: <input type="text" name="ramto" value="<?php echo "$cookie_ramto"; ?>" size="8"></td>
</td>
</tr>
<td><b>Current Uptime (in days)</b></td> 
<td>From: <input type="text" name="uptimefrom" value="<?php echo "$cookie_uptimefrom"; ?>" size="8">
 To: <input type="text" name="uptimeto" value="<?php echo "$cookie_uptimeto"; ?>" size="8"></td>
</td>
</tr>
<td><b>Total Uptime (in days)</b></td> 
<td>From: <input type="text" name="totuptimefrom" value="<?php echo "$cookie_totuptimefrom"; ?>" size="8">
 To: <input type="text" name="totuptimeto" value="<?php echo "$cookie_totuptimeto"; ?>" size="8"></td>
</td>
</tr>
</table>
<input type="Submit">  <a href="search_cancel.php">Cancel Search</a><br><br>
<p>
<a href="list_clients.php">Back to List Clients</a>
</p>
</form>
</body>
</html>
