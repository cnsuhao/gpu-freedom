
CREATE TABLE `tbclient` (
`id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY ,
`nodeid` VARCHAR( 32 ) NOT NULL ,
`nodename` VARCHAR( 32 ) NOT NULL ,
`country` VARCHAR( 32 ) NOT NULL ,
`region` VARCHAR( 32 ) NULL ,
`city` VARCHAR( 32 ) NULL ,
`zip` VARCHAR( 32 ) NULL ,
`ip` VARCHAR( 32 ) NULL ,
`port` VARCHAR( 32 ) NULL ,
`localip` VARCHAR( 32 ) NULL ,
`os` VARCHAR( 32 ) NOT NULL ,
`version` VARCHAR( 16 ) NOT NULL,
`acceptincoming` INT NOT NULL DEFAULT '0',
`gigaflops` INT NOT NULL,
`ram` INT NOT NULL,
`mhz` INT NOT NULL,
`nbcpus` INT NOT NULL,
`bits` INT NOT NULL,
`isscreensaver` INT NOT NULL DEFAULT '0',
`uptime` DOUBLE NOT NULL ,
`totaluptime` DOUBLE NOT NULL ,
`longitude` DOUBLE NOT NULL ,
`latitude` DOUBLE NOT NULL ,
`userid` VARCHAR( 32 ) NOT NULL ,
`team` VARCHAR( 64 ) NOT NULL ,
`description` VARCHAR( 256 ) NULL ,
`cputype` VARCHAR( 64 ) NULL,
`create_dt` DATETIME NOT NULL,
`update_dt` DATETIME NULL
) ENGINE = MYISAM ;


CREATE TABLE `tbchannel` (
`id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY ,
`nodeid` VARCHAR( 32 ) NOT NULL ,
`nodename` VARCHAR( 32 ) NOT NULL ,
`user` VARCHAR( 32 ) NOT NULL ,
`channame` VARCHAR( 32 ) NOT NULL ,
`chantype` VARCHAR( 32 ) NOT NULL ,
`content` VARCHAR( 1024 ) NOT NULL ,
`ip` VARCHAR( 32 ) NULL ,
`usertime_dt` DATETIME NULL,
`create_dt` DATETIME NOT NULL
) ENGINE = MYISAM ;


CREATE TABLE `tbparameter` (
  `id` int(11) NOT NULL auto_increment,
  `paramtype` varchar(20) collate latin1_general_ci NOT NULL,
  `paramname` varchar(20) collate latin1_general_ci NOT NULL,
  `paramvalue` varchar(255) collate latin1_general_ci NOT NULL,
  PRIMARY KEY  (`id`),
  UNIQUE KEY `paramtype` (`paramtype`,`paramname`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 COLLATE=latin1_general_ci AUTO_INCREMENT=3 ;


INSERT INTO `tbparameter` (`id`, `paramtype`, `paramname`, `paramvalue`) VALUES
(1, 'TEST', 'DB_CONNECTION', 'OK');

CREATE TABLE `tbjob` (
`id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY ,
`jobid` VARCHAR( 16 ) NOT NULL ,
`job` VARCHAR( 1024 ) NOT NULL ,
`workunitincoming` VARCHAR( 64 ) NOT NULL ,
`workunitoutgoing` VARCHAR( 64 ) NOT NULL ,
`requests` INT NOT NULL DEFAULT '1',
`delivered` INT NOT NULL DEFAULT '0',
`results` INT NOT NULL DEFAULT '0',
`nodename` VARCHAR( 64 ) NOT NULL ,
`nodeid` VARCHAR( 32 ) NOT NULL ,
`ip` VARCHAR( 32 ) NULL ,
`create_dt` DATETIME NOT NULL
) ENGINE = MYISAM ;

CREATE TABLE `tbjobqueue` (
`id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY ,
`job_id` INT NOT NULL ,
`nodeid` VARCHAR( 32 ) NOT NULL ,
`transmitted` INT NOT NULL DEFAULT '0',
`received` INT NOT NULL DEFAULT '0',
`create_dt` DATETIME NOT NULL,
`transmission_dt` DATETIME NULL,
`reception_dt` DATETIME NULL
) ENGINE = MYISAM ;

CREATE TABLE `tbjobresult` (
`id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY ,
`job_id` INT NOT NULL ,
`jobqueue_id` INT NOT NULL ,
`jobresult` VARCHAR( 1024 ) NOT NULL ,
`workunitresult` VARCHAR( 64 ) NOT NULL ,
`iserroneous` INT NOT NULL DEFAULT '0',
`errorid` INT NOT NULL DEFAULT '0',
`errorarg` VARCHAR( 32 ) NOT NULL ,
`errormsg` VARCHAR( 32 ) NOT NULL ,
`nodename` VARCHAR( 64 ) NOT NULL ,
`nodeid` VARCHAR( 32 ) NOT NULL ,
`ip` VARCHAR( 32 ) NULL ,
`create_dt` DATETIME NOT NULL
) ENGINE = MYISAM ;
