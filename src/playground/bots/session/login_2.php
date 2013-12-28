<?php session_start(); ?>
<html>
<body>
<?php
include("login.inc.php");
$name=mysql_real_escape_string($_POST['name']);
$pwd=mysql_real_escape_string($_POST['pwd']);
If ($name=="") exit;

if (($name==$username) && ($pwd==$password)) {
    echo "<b>login ok :-)</b>";
	$_SESSION['userbot'] = $name;
	$_SESSION['userpwd'] = $pwd;
	
	js_redirect("../bitcoin/index.php");
} else {
    $_SESSION['userbot'] = "";
	$_SESSION['userpwd'] = "";
	die("<b>login failed</b>");
}

?>
</body>
</html>