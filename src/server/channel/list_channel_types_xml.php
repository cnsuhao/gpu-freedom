<?php include('../utils/sql2xml.php') ?>
 
<channeltype>
<?php 
$level_list = Array("type");
sql2xml('select distinct c.channame from tbchannel c', $level_list, 0);
?>
</channeltype>
