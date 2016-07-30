CREATE TABLE IF NOT EXISTS `tbtemperature` (
  `insert_dt` datetime NOT NULL,
  `temperature_raspi` float NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;



INSERT INTO `tbtemperature` (`insert_dt`, `temperature_raspi`) VALUES
('2016-07-27 14:26:49', 56.4);
