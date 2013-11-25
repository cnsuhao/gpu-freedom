<?php session_start(); ?>
<html>
<body>
<?php
include("config.inc.php");
$name=mysql_real_escape_string($_POST['name']);
$pwd=mysql_real_escape_string($_POST['pwd']);
If ($name=="") exit;

if (($name==$username) && ($pwd==$password)) {
    echo "<b>login ok :-)</b>";
	$_SESSION['userbot'] = $name;
} else {
    $_SESSION['userbot'] = "";
	die("<b>login failed</b>");
}

?>
</body>
</html>