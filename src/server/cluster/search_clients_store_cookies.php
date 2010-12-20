<?php
   include("../utils/utils.inc.php");
   
   $nodename=$_POST['nodename'];
   $cputype=$_POST['cputype'];
   $os=$_POST['os'];
   $country=$_POST['country'];
   $mhzfrom=$_POST['mhzfrom'];
   $mhzto=$_POST['mhzto'];
   $ramfrom=$_POST['ramfrom'];
   $ramto=$_POST['ramto'];
   $uptimefrom=$_POST['uptimefrom'];
   $uptimeto=$_POST['uptimeto'];
   $totuptimefrom=$_POST['totuptimefrom'];
   $totuptimeto=$_POST['totuptimeto'];
   $fromversion=$_POST['fromversion'];
   $toversion=$_POST['toversion'];
   $onlyonline=$_POST['onlyonline'];
   
   $entriesoffset = 0;
   js_cookie("entriesoffset", $entriesoffset);
  
   js_cookie("search_nodename", $nodename);
   js_cookie("search_os", $os);
   
   js_cookie("search_cputype", $cputype);
   js_cookie("search_country",   $country);
     
   js_cookie("search_mhzfrom", $mhzfrom);
   js_cookie("search_mhzto", $mhzto);
   js_cookie("search_ramfrom", $ramfrom);
   js_cookie("search_ramto", $ramto);
   js_cookie("search_uptimefrom", $uptimefrom);
   js_cookie("search_uptimeto", $uptimeto);
   js_cookie("search_totuptimefrom", $totuptimefrom);
   js_cookie("search_totuptimeto", $totuptimeto);
   
   js_cookie("search_fromversion", $fromversion);
   js_cookie("search_toversion", $toversion);

   js_redirect("list_clients.php");
?>