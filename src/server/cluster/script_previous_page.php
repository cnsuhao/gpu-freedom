<?php
 include("utils/constants.inc.php");
 include("utils/utils.inc.php");
 
 $entryoffset = $_COOKIE["entryoffset"];
 if ($entryoffset=="") $entryoffset=0;
 $entryoffset-=$entriesperpage;
 if ($entryoffset<0) $entryoffset=0;
 js_cookie("entryoffset", $entryoffset);
 js_redirect("list_computers.php");
?>