<?php
 include("utils/constants.inc.php");
 include("utils/utils.inc.php");
 
 $entryoffset = 0;
 js_cookie("entryoffset", $entryoffset);
 js_redirect("list_computers.php");
?>