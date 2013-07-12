<?php
include_once( '../utils/openflashchart/open-flash-chart.php' );
include("../conf/config.inc.php");

 // preparing data for chart
 mysql_connect($dbserver, $username, $password);
 @mysql_select_db($database) or die("Unable to select database");
 $query="select networkdiff from tbfrequency where (create_dt>NOW() - INTERVAL 30 DAY) order by create_dt asc LIMIT 1000;";
 $result=mysql_query($query);
 
 if ($result=="") {
    mysql_close();
    exit();
 } else {
  $num=mysql_numrows($result); 
 }
 
 $data = array();
 $labels = array();
 $i=0;
 $max=100;
 while ($i<$num) { 
   $net=mysql_result($result,$i,"networkdiff");          
   
   $data[$i] = $net;
   $labels[$i] = '';
   
   if ($freq>$max) $max=$freq;
   
   $i++;
 }
 mysql_close();
 


// use the chart class to build the chart:
$g = new graph();
$g->title( 'Network/Real time diff (s)', '{font-size:18px; color: #d01f3c}' );

//
// pass in two arrays, one of data, the other data labels
//
$g->set_data($data);
$g->set_x_labels($labels);
$g->set_x_label_style( 10, '#9933CC', 0, 2 );
$g->set_y_max(45);
$g->set_y_min(-45);


$g->set_tool_tip( '#val#' );


// display the data
echo $g->render();
?>