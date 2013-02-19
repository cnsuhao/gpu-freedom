<?php include('../utils/sql2xml.php') ?>
 
<channel>
<?php sql2xml('select c.id, c.nodeid from tbchannel c', '0') ?>
</channel>
