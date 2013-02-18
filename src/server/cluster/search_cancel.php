<?php
 include("../utils/utils.inc.php");
   
 $entriesoffset = 0;
 js_cookie("entriesoffset", $entriesoffset);
  
 js_cookie("search_nodename", "");
 js_cookie("search_cputype", "");
   
 js_cookie("search_os", "");
 js_cookie("search_country",   "");
     
 js_cookie("search_mhzfrom", "");
 js_cookie("search_mhzto", "");
 js_cookie("search_ramfrom", "");
 js_cookie("search_ramto", "");
 js_cookie("search_uptimefrom", "");
 js_cookie("search_uptimeto", "");
 js_cookie("search_totuptimefrom", "");
 js_cookie("search_totuptimeto", "");
   
 js_cookie("search_fromversion", "");
 js_cookie("search_toversion", "");

 js_redirect("list_clients.php");

?>