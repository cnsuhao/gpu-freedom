<?php

function getparam($name, $default) {
 if (isset($_GET["$name"])) return $_GET["$name"]; else return $default;
}

function js_redirect($s)
    {
        print "<body onload=\"window.location='$s';\">";
        print "<a href='$s'>Javascript redirect.. If your page doesn't redirect click here.</a>";
        print "</body>";
        exit();
}

function js_cookie($name,$value)
{
        print "<script>";
        print "document.cookie='$name=$value'";
        print "</script>";
}

function get_parameter_default($paramtype, $paramname, $userid, $defaultparam) {
  if ($userid=="") $userquery="AND user_id IS NULL";
  else $userquery="AND user_id=$userid";
  
  $query="SELECT paramvalue from tbparameter where paramtype='$paramtype' and paramname='$paramname' $userquery"; 
  $result=mysql_query($query);  
  $num=mysql_numrows($result);
  
  if ($num>0) 
     $paramvalue=mysql_result($result,0,"paramvalue");
  else
     $paramvalue=$defaultparam;
  return $paramvalue;
}

function get_parameter($paramtype, $paramname, $userid) {
  $paramvalue=get_parameter_default($paramtype,$paramname,$userid,"");
  return $paramvalue;
}

function set_parameter($paramtype, $paramname, $paramvalue, $userid) {
  if ($userid=="") $userupdate="";
  else $userupdate="AND user_id=$userid";
  
  // decide first if we need to insert a new parameter
  $check = get_parameter_default($paramtype, $paramname, $userid, 'missing parameter');
  if ($check=='missing parameter') {
      if ($userid=="") $userid="NULL";
	  $query="INSERT INTO tbparameter (id, paramtype, paramname, paramvalue, user_id) VALUES('', '$paramtype', '$paramname', '$paramvalue', $userid);";    	  
  }
  else
      $query="UPDATE tbparameter p SET p.paramvalue='$paramvalue' where paramtype='$paramtype' and paramname='$paramname' $userupdate"; 
  mysql_query($query); 
}

function retrieve_salt() {
  return get_parameter('SECURITY','PWD_HASH_SALT', '');
}

function salt_and_hash($pwd, $salt) {  
  return md5("$pwd$salt");
}

?>