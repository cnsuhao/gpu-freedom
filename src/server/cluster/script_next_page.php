<?php
 include("../utils/utils.inc.php");
 
 $entryoffset = $_COOKIE["entryoffset"];
 if ($entryoffset=="") $entryoffset=0;
 $entryoffset+=$entriesperpage;
 js_cookie("entryoffset", $entryoffset);
 js_redirect("list_clients.php");
?>