<?php
 include("utils/constants.inc.php");
 include("utils/utils.inc.php");

 $count=$_GET['nbentries'];
 $entryoffset=$count - ($count % $entriesperpage);
 js_cookie("entryoffset", $entryoffset);
 js_redirect("list_computers.php");

?>