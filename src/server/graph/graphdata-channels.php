<?php

include_once( '../utils/openflashchart/open-flash-chart.php' );
include("../conf/config.inc.php");
include("../utils/utils.inc.php");
if (getPHPVersion()>=50500) include_once('../utils/mydql2i/mysql2i.class.php');

 // preparing data for chart
 mysql_connect($dbserver, $username, $password);
 @mysql_select_db($database) or die("Unable to select database");
 $query="SELECT count(*) , c.channame FROM tbchannel c GROUP BY channame ORDER BY c.channame LIMIT 0,30";
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
 while ($i<$num) { 
   $module=mysql_result($result,$i,"channame");
   $count=mysql_result($result,$i,"count(*)");          
   
   $data[$i] = $module;
   $labels[$i] = $count;
   
   $i++;
 }
 mysql_close();
 


// use the chart class to build the chart:
$g = new graph();
$g->title( 'Channel Size', '{font-size:18px; color: #d01f3c}' );

//
// PIE chart, 60% alpha
//
$g->pie(60,'#505050','{font-size: 12px; color: #404040;');
//
// pass in two arrays, one of data, the other data labels
//
$g->pie_values( $labels, $data);
//
// Colours for each slice, in this case some of the colours
// will be re-used (3 colurs for 5 slices means the last two
// slices will have colours colour[0] and colour[1]):
//
$g->pie_slice_colours( array('#d01f3c','#356aa0','#C79810') );

$g->set_tool_tip( '#val#' );


// display the data
echo $g->render();
?>