<?php
	include('../utils/constants.inc.php');
	include('../utils/utils.inc.php');
	include('geoip.inc.php');
	
	
	$ip = getparam('ip', false); 
	if (!$ip) { 
		// if the ip is not provided, we use the external ip of thecaller
		$ip = $_SERVER['SERVER_ADDR'];
		if (($ip=="127.0.0.1") || (substr($ip,0,7)=="192.168") || (substr($ip,0,3)=="10.")) {
			echo "<geoip>\n";
			echo "   <ip>$ip</ip>\n";
			echo "   <error>Could not retrieve external ip address</error>\n";
			echo "<geoip>\n";
			die("");
		}
	}
	
		$resarray = get_geoip_info($ip);
		echo "<geoip>\n";
		echo "   <longitude>";
		echo $resarray["location"]["longitude"];
		echo "</longitude>\n";
		echo "   <latitude>";
		echo $resarray["location"]["latitude"];
		echo "</latitude>\n";
		echo "   <city>";
		echo $resarray["city"];
		echo "</city>\n";
        echo "   <countryname>";
		echo $resarray["country"]["name"];
		echo "</countryname>\n";
	    echo "   <countrycode>";
		echo $resarray["country"]["code"];
		echo "</countrycode>\n";
	    echo "   <timezone>";
		echo $resarray["location"]["time_zone"];
		echo "</timezone>\n";
		echo "   <ip>";
		echo $resarray["ip"];
		echo "</ip>\n";
	    echo "</geoip>\n";

?>