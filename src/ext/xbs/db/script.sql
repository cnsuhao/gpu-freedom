SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";

--
-- Datenbank: `xbs`
--

-- --------------------------------------------------------

CREATE TABLE IF NOT EXISTS `tbparameter` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `paramtype` varchar(20) COLLATE latin1_general_ci NOT NULL,
  `paramname` varchar(20) COLLATE latin1_general_ci NOT NULL,
  `paramvalue` varchar(255) COLLATE latin1_general_ci NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `paramtype` (`paramtype`,`paramname`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 COLLATE=latin1_general_ci AUTO_INCREMENT=4 ;


INSERT INTO `tbparameter` (`id`, `paramtype`, `paramname`, `paramvalue`) VALUES
(1, 'ESS', 'VERSION', '3.3');

CREATE TABLE IF NOT EXISTS `counterparty` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(32) COLLATE latin1_general_ci NOT NULL,
  `legalname` varchar(64) COLLATE latin1_general_ci NOT NULL,
  `iscompany` tinyint(1) NOT NULL DEFAULT '0',
  `eiccode` varchar(16) COLLATE latin1_general_ci NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 COLLATE=latin1_general_ci AUTO_INCREMENT=2 ;


CREATE TABLE IF NOT EXISTS `controlarea` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(32) COLLATE latin1_general_ci NOT NULL,
  `region` varchar(32) COLLATE latin1_general_ci NOT NULL,
  `eiccode` varchar(16) COLLATE latin1_general_ci NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 COLLATE=latin1_general_ci AUTO_INCREMENT=2 ;

-- company_id, can reference only counterparties with iscompany=1
-- counterparty_id, can reference only counterparties with iscompany=0
CREATE TABLE IF NOT EXISTS `contract` (
  `id` int(11) NOT NULL AUTO_INCREMENT, 
  `name` varchar(32) COLLATE latin1_general_ci NOT NULL,
  `company_id` int(11) DEFAULT NULL,
  `counterparty_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 COLLATE=latin1_general_ci AUTO_INCREMENT=2 ;


-- direction: 1 BUY, 0 TRANSMISSION, -1 SELL
-- quantitytype: 1 Standard (requires basepeakoff and quantity), 2 Profile (requires profile_id), 3 Block (requires block_id)
-- basepeakoff: 0 not set, 1 Base, 2 Peak, 3 Offpeak
-- pricetype: 1 Fix, 2 Indexed (requires index_id)
-- currency: EUR or CHF or similar
CREATE TABLE IF NOT EXISTS `powertrade` (
  `id` int(11) NOT NULL AUTO_INCREMENT, 
  `contract_id` int(11) DEFAULT NULL,
  `tradedate` datetime NOT NULL,
  `begintime` datetime NOT NULL,
  `endtime` datetime NOT NULL, 
  `direction` int(11) NOT NULL,
  `areafrom_id` int(11) NOT NULL,
  `areato_id` int(11) NOT NULL,
  `quantitytype` int(11) NOT NULL,
  `quantity` int(11) DEFAULT NULL,
  `basepeakoff` int(11) DEFAULT '0',
  `profile_id` int(11) DEFAULT NULL,
  `block_id` int(11) DEFAULT NULL,
  `pricetype` int(11)  NOT NULL,
  `priceindex_id` int(11) DEFAULT NULL,
  `price` int(11) DEFAULT NULL,
  `currency` varchar(3) COLLATE latin1_general_ci NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 COLLATE=latin1_general_ci AUTO_INCREMENT=2 ;

