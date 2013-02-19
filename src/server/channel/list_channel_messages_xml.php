<?php include('../utils/sql2xml.php') ?>
 
<channel>
<?php 
$level_list = Array("msg");
sql2xml('select c.id, c.content, c.nodeid, c.nodename, c.user, c.channame, c.chantype, c.usertime_dt, c.create_dt from tbchannel c', $level_list, 0);
?>
</channel>
