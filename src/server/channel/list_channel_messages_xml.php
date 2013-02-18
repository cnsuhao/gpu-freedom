<?php include('../utils/sql2xml.php') 
	  include('../conf/config.inc.php')
?>
 
<channel>
<?php sql2xml('select a.alb_id, a.alb_name,
               s.sng_number, s.sng_name
               from album a, song s
               where
               a.alb_id = s.alb_id and
               s.sng_number < 4
               order by a.alb_id, s.sng_number', '2') ?>
</channel>
?>