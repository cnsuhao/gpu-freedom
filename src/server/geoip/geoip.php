<?php
	include('../utils/constants.inc.php');
	include('../utils/utils.inc.php');
	include('geoip.inc.php');
	
	
	$ip = getparam('ip', false); 
	
	if ($ip) {
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
    }

?>