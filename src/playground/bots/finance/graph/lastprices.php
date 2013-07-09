<?php
include_once( '../utils/openflashchart/open-flash-chart.php' );
include("../conf/config.inc.php");
if (isset($_GET['ticker'])) $ticker=$_GET['ticker']; else die("<b>Ticker Parameter not defined</b>");

 // preparing data for chart
 mysql_connect($dbserver, $username, $password);
 @mysql_select_db($database) or die("Unable to select database");
 $query="select value from tickers where name='$ticker' AND (create_dt>NOW() - INTERVAL 30 DAY) order by create_dt asc LIMIT 300;";
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
   $price=mysql_result($result,$i,"value");          
   
   $data[$i] = $price;
   $labels[$i] = '';
   
   if ($price>$max) $max=$price;
   
   $i++;
 }
 
 // retrieve description of the ticker
 $querydesc="select description from tickernames where name='$ticker';";
 $resultdesc=mysql_query($querydesc);
 $description=mysql_result($resultdesc, 0, "description");
 
 mysql_close();
 


// use the chart class to build the chart:
$g = new graph();
$g->title( "$description ($ticker)", '{font-size:18px; color: #d01f3c}' );

//
// pass in two arrays, one of data, the other data labels
//
$g->set_data($data);
$g->set_x_labels($labels);
$g->set_x_label_style( 10, '#9933CC', 0, 2 );
$g->set_y_max($max);

$g->set_tool_tip( '#val#' );


// display the data
echo $g->render();
?>