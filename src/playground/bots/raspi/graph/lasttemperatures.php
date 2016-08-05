<?php
include_once('../../../../server/utils/mydql2i/mysql2i.class.php');
include_once( '../utils/openflashchart/open-flash-chart.php' );
include("../conf/config.inc.php");

 // preparing data for chart
 mysql_connect($dbserver, $username, $password);
 @mysql_select_db($database) or die("Unable to select database");
 $query="select temperature_raspi from tbtemperature where (insert_dt>NOW() - INTERVAL 30 DAY) order by insert_dt asc LIMIT 3000;";
 $result=mysql_query($query);
 
 if ($result=="") {
    mysql_close();
    exit();
 } else {
  $num=mysql_num_rows($result); 
 }
 
 $data = array();
 $labels = array();
 $i=0;
 $max=150; // probably Raspi3 dies here!!!
 while ($i<$num) { 
   $temp=mysql_result($result,$i,"temperature_raspi");          
   
   $data[$i] = $temp;
   $labels[$i] = '';
   
   if ($temp>$max) $max=$temp;
   
   $i++;
 }
 mysql_close();
 


// use the chart class to build the chart:
$g = new graph();
$g->title( 'Temperatures (Celsius)', '{font-size:18px; color: #d01f3c}' );

//
// pass in two arrays, one of data, the other data labels
//
$g->set_data($data);
$g->set_x_labels($labels);
$g->set_x_label_style( 10, '#9933CC', 0, 2 );
$g->set_y_max(95);
$g->set_y_min(15);


$g->set_tool_tip( '#val#' );


// display the data
echo $g->render();
?>
