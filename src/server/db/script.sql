
CREATE TABLE `tbproject` (
`id` INT NOT NULL AUTO_INCREMENT ,
`name` VARCHAR( 32 ) NOT NULL ,
`folder` VARCHAR( 16 ) NOT NULL ,
`description` VARCHAR( 254 ) ,
`isreportresult` INT DEFAULT '1',
`isupload` INT DEFAULT '0',
`nb_passes` INT DEFAULT '1' NOT NULL ,
PRIMARY KEY ( `id` ) ,
UNIQUE (
`folder` 
)
);
ALTER TABLE `tbproject` ADD `status` VARCHAR( 32 ) NOT NULL DEFAULT 'None';
ALTER TABLE `tbproject` ADD `current_pass` INT NOT NULL DEFAULT '0' AFTER `nb_passes` ;
ALTER TABLE `tbproject` ADD `tot_requests` INT NOT NULL DEFAULT '0';
ALTER TABLE `tbproject` ADD `tot_results` INT NOT NULL DEFAULT '0';
ALTER TABLE `tbproject` ADD `owner` VARCHAR( 32 ) NOT NULL DEFAULT 'None';
ALTER TABLE `tbproject` ADD `isexecutable` INT( 11 ) NOT NULL DEFAULT '0' AFTER `isupload` ;
ALTER TABLE `tbproject` ADD `issinglewu` INT NOT NULL DEFAULT '0',
ADD `singlewuname` VARCHAR( 128 ) NULL ,
ADD `size` INT NULL ;
ALTER TABLE `tbproject` ADD `isforcedistribution` INT NOT NULL DEFAULT '0';

CREATE TABLE `tbwork` (
`id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY ,
`filename` VARCHAR( 254 ) NOT NULL ,
`project_id` INT NOT NULL ,
`requests` INT NOT NULL DEFAULT '0',
`results` INT NOT NULL DEFAULT '0',
`result_1` VARCHAR( 254 ) NULL ,
`result_2` VARCHAR( 254 ) NULL ,
`result_3` VARCHAR( 254 ) NULL ,
`result_4` VARCHAR( 254 ) NULL ,
`result_5` VARCHAR( 254 ) NULL ,
`status` VARCHAR( 16 ) NOT NULL DEFAULT 'None'
) ENGINE = MYISAM ;
ALTER TABLE `tbwork` ADD `processor_1` VARCHAR( 64 ) NULL ;
ALTER TABLE `tbwork` ADD `processor_2` VARCHAR( 64 ) NULL ;
ALTER TABLE `tbwork` ADD `processor_3` VARCHAR( 64 ) NULL ;
ALTER TABLE `tbwork` ADD `processor_4` VARCHAR( 64 ) NULL ;
ALTER TABLE `tbwork` ADD `processor_5` VARCHAR( 64 ) NULL ;
ALTER TABLE `tbwork` ADD `workunitnb` INT NULL AFTER `id` ;

CREATE TABLE `users` (
  `id` int(11) NOT NULL auto_increment,
  `username` varchar(32) collate latin1_general_ci NOT NULL,
  `password` varchar(32) collate latin1_general_ci default NULL,
  `first` varchar(32) collate latin1_general_ci default NULL,
  `last` varchar(32) collate latin1_general_ci default NULL,
  `email` varchar(64) collate latin1_general_ci default NULL,
  `rights` int(11) NOT NULL default '0',
  PRIMARY KEY  (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=MyISAM;
ALTER TABLE `users` ADD `team_id` INT NULL ;

INSERT INTO `users` ( `id` , `username` , `password` , `first` , `last` , `email` , `rights` )
VALUES (NULL , 'admin', 'gpufd', 'Main', 'Administrator', 'admin@gpufd', '3');


CREATE TABLE `tbexecutable` (
`id` INT( 11 ) NOT NULL AUTO_INCREMENT PRIMARY KEY ,
`project_id` INT( 11 ) NOT NULL ,
`win_x86` VARCHAR( 255 ) NULL ,
`linux_x86` VARCHAR( 255 ) NULL ,
`linux_ppc` VARCHAR( 255 ) NULL ,
`macosx_x86` VARCHAR( 255 ) NULL ,
`macosx_ppc` VARCHAR( 255 ) NULL ,
INDEX ( `project_id` )
) ENGINE = MYISAM ;
ALTER TABLE `tbexecutable` ADD `extractfolder` VARCHAR( 128 ) NULL ;

CREATE TABLE `tbftpupload` (
  `id` int(11) NOT NULL auto_increment,
  `project_id` int(11) NOT NULL,
  `ftpurl` varchar(255) collate latin1_general_ci NOT NULL,
  `ftppath` varchar(255) collate latin1_general_ci NOT NULL,
  `prefix` varchar(64) collate latin1_general_ci default NULL,
  PRIMARY KEY  (`id`),
  KEY `project_id` (`project_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_general_ci AUTO_INCREMENT=1 ;
ALTER TABLE `tbftpupload` ADD `ftpport` INT( 11 ) NOT NULL DEFAULT '21',
ADD `user` VARCHAR( 64 ) NULL ,
ADD `password` VARCHAR( 64 ) NULL ;
ALTER TABLE `tbftpupload` ADD `extension` VARCHAR( 64 ) NULL AFTER `prefix` ;

CREATE TABLE `tbprocessor` (
`id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY ,
`processor` VARCHAR( 128 ) NOT NULL ,
`user_id` INT NULL ,
`description` VARCHAR( 256 ) NULL ,
`cputype` VARCHAR( 64 ) NULL
) ENGINE = MYISAM ;
ALTER TABLE `tbprocessor` ADD `mhz` INT NULL ,
ADD `ram` INT NULL ,
ADD `cpus` INT NULL ;
ALTER TABLE `tbprocessor` ADD `operatingsystem` VARCHAR( 128 ) NULL ;
ALTER TABLE `tbprocessor` ADD `team_id` INT NULL ,
ADD `uptime` DOUBLE NULL ,
ADD `totuptime` DOUBLE NULL ,
ADD `zip` VARCHAR( 16 ) NULL ,
ADD `city` VARCHAR( 64 ) NULL ,
ADD `region` VARCHAR( 64 ) NULL ,
ADD `country` VARCHAR( 32 ) NULL ,
ADD `geolocation_x` DOUBLE NULL ,
ADD `geolocation_y` DOUBLE NULL ;
ALTER TABLE `tbprocessor` ADD `nodeid` VARCHAR( 64 ) NULL AFTER `user_id` ,
ADD `ip` VARCHAR( 32 ) NULL AFTER `nodeid` ,
ADD `port` INT NULL AFTER `ip` ,
ADD `acceptincoming` INT NOT NULL DEFAULT '0' AFTER `port` ,
ADD `updated` DATETIME NULL AFTER `acceptincoming` ;
ALTER TABLE `tbprocessor` ADD `freeconn` INT NULL ,
ADD `maxconn` INT NULL ;
ALTER TABLE `tbprocessor` ADD `version` VARCHAR( 16 ) NULL;


CREATE TABLE `tbgpuprocessor` (
`id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY ,
`processor_id` INT NOT NULL ,
`abarth` VARCHAR( 16 ) NULL ,
`speed` INT NULL ,
`crawlo` INT NULL ,
`terra` INT NULL ,
`threads` INT NULL ,
`inqueue` INT NULL ,
`trafficdown` INT NULL ,
`trafficup` INT NULL ,
`partlevel` DOUBLE NULL ,
`ip1` VARCHAR( 32 ) NULL ,
`ip2` VARCHAR( 32 ) NULL ,
`ip3` VARCHAR( 32 ) NULL ,
`ip4` VARCHAR( 32 ) NULL ,
`ip5` VARCHAR( 32 ) NULL ,
`ip6` VARCHAR( 32 ) NULL ,
`ip7` VARCHAR( 32 ) NULL ,
`ip8` VARCHAR( 32 ) NULL ,
`ip9` VARCHAR( 32 ) NULL ,
`ip10` VARCHAR( 32 ) NULL
) ENGINE = MYISAM ;

ALTER TABLE `tbgpuprocessor`
ADD `updated` DATETIME NULL;
ALTER TABLE `tbgpuprocessor` 
ADD `ips` INT( 11 ) NOT NULL AFTER `partlevel` ;
ALTER TABLE `tbgpuprocessor` 
ADD `listenip` VARCHAR( 32 ) NULL AFTER `trafficup`; 


CREATE TABLE `tbteam` (
`id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY ,
`name` VARCHAR( 64 ) NOT NULL ,
`description` VARCHAR( 1024 ) NULL ,
`url` VARCHAR( 256 ) NULL
) ENGINE = MYISAM ;








