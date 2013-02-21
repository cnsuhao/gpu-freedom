<servers>
 <server>
    <externalid>1</externalid>
    <servername>Altos</servername>
    <serverurl>http://www.gpu-grid.net/superserver</serverurl>
    <chatchannel>altos</chatchannel>
    <version>1.0.0</version>
    <superserver>true</superserver>
    <uptime>1.3</uptime>
    <totaluptime>15.24</totaluptime>
    <longitude>6</longitude>
    <latitude>-70</latitude>
    <activenodes>13</activenodes>
    <jobsinqueue>3</jobsinqueue>
 </server>

 <server>
    <externalid>2</externalid>
    <servername>Orion</servername>
    <serverurl>http://www.gpu-grid.net/testserver</serverurl>
    <chatchannel>orion</chatchannel>
    <version>1.0.0</version>
    <superserver>false</superserver>
    <uptime>0.2</uptime>
    <totaluptime>10.6</totaluptime>
    <longitude>-150</longitude>
    <latitude>17</latitude>
    <activenodes>7</activenodes>
    <jobsinqueue>12</jobsinqueue>
 </server>


</servers>


CREATE TABLE IF NOT EXISTS `tbserver` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `serverid` varchar(42) NOT NULL,
  `servername` varchar(32) NOT NULL,
  `serverurl`  varchar(256) NOT NULL,
  `chatchannel` varchar(32) NULL,
  `version`    double NOT NULL,
  `superserver`    BOOL NOT NULL,
  `ip` varchar(32) DEFAULT NULL,
  `uptime` double NOT NULL,
  `totaluptime` double NOT NULL,
  
  
  `create_dt` datetime NOT NULL,
  `update_dt` datetime NULL
  PRIMARY KEY (`id`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 AUTO_INCREMENT=4 ;