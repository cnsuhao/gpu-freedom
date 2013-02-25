<?php

function get_db_parameter($paramtype, $paramname, $defaultparam) {
 
  $query="SELECT paramvalue from tbparameter where paramtype='$paramtype' and paramname='$paramname' LIMIT 1;"; 
  $result=mysql_query($query);  
  $num=mysql_numrows($result);
  
  if ($num>0) 
     $paramvalue=mysql_result($result,0,"paramvalue");
  else
     $paramvalue=$defaultparam;
  return $paramvalue;
}


function set_db_parameter($paramtype, $paramname, $paramvalue) {
  
  // decide first if we need to insert a new parameter
  $check = get_db_parameter($paramtype, $paramname, 'missing parameter');
  if ($check=='missing parameter') {
 	  $query="INSERT INTO tbparameter (id, paramtype, paramname, paramvalue) VALUES('', '$paramtype', '$paramname', '$paramvalue');";    	  
  }
  else
      $query="UPDATE tbparameter p SET p.paramvalue='$paramvalue' where paramtype='$paramtype' and paramname='$paramname';"; 
  mysql_query($query); 
}


?>