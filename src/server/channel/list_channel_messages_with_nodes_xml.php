<?php include('../utils/sql2xml.php') ?>
 
<channel>
<?php 
$level_list = Array("msg", "client");
sql2xml('select c.id, c.content, c.nodeid, c.nodename, c.user, c.channame, c.chantype, c.usertime_dt, c.create_dt,
         cl.country, cl.longitude, cl.latitude 
         from tbchannel c, tbclient cl 
         where c.nodeid=cl.nodeid'
		 , $level_list, 9);
?>
</channel>
