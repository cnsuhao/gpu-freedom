<?php

include_once( '../utils/openflashchart/open-flash-chart.php' );
include("../conf/config.inc.php");

 // preparing data for chart
 mysql_connect($dbserver, $username, $password);
 @mysql_select_db($database) or die("Unable to select database");
 $query="select volume from pricevalue where (create_dt>NOW() - INTERVAL 10 DAY) order by create_dt asc LIMIT 20000;";
 $result=mysql_query($query);
 
 if ($result=="") {
    mysql_close();
    exit();
 } else {
  $num=mysql_numrows($result); 
 }

 $bar = new bar_outline( 50, '#9933CC', '#8010A0' );
 $bar->key( 'Volume', 10 );
 
 $data = array();
 $labels = array();
 $i=0;
 $max=0;
 while ($i<$num) { 
   // my is for month-year
   $volume=mysql_result($result,$i,"volume");          
   
   $bar->data[] = $volume;
   $labels[$i] = '';
   
   if ($volume>$max) $max=$volume;
   
   $i++;
 }
 mysql_close();
 
 //$bar->$data[] = $data[];

// use the chart class to build the chart:
$g = new graph();
$g->title( 'Volume', '{font-size:18px; color: #d01f3c}' );

//
// pass in two arrays, one of data, the other data labels
//
$g->data_sets[] = $bar;
$g->set_x_labels($labels);
$g->set_x_label_style( 10, '#9933CC', 0, 2 );
$g->set_y_max($max);

$g->set_tool_tip( '#val#' );


// display the data
echo $g->render();
?>