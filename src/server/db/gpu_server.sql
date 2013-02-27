-- phpMyAdmin SQL Dump
-- version 3.5.2
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Feb 27, 2013 at 05:58 PM
-- Server version: 5.5.25a
-- PHP Version: 5.4.4

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `gpu_server`
--

DELIMITER $$
--
-- Functions
--
CREATE DEFINER=`root`@`localhost` FUNCTION `PLANAR_DISTANCE`(lat1 DOUBLE, lon1 DOUBLE, lat2 DOUBLE,  lon2 DOUBLE) RETURNS double
    DETERMINISTIC
BEGIN
     DECLARE dist DOUBLE;
     SET dist = SQRT((lat1-lat2)*(lat1-lat2)+(lon1-lon2)*(lon1-lon2));
     RETURN dist;
    END$$

DELIMITER ;

-- --------------------------------------------------------

--
-- Table structure for table `tbchannel`
--

CREATE TABLE IF NOT EXISTS `tbchannel` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `nodeid` varchar(42) NOT NULL,
  `nodename` varchar(32) NOT NULL,
  `user` varchar(32) NOT NULL,
  `channame` varchar(32) NOT NULL,
  `chantype` varchar(32) NOT NULL,
  `content` varchar(1024) NOT NULL,
  `ip` varchar(32) DEFAULT NULL,
  `usertime_dt` datetime DEFAULT NULL,
  `create_dt` datetime NOT NULL,
  PRIMARY KEY (`id`),
  KEY `nodeid_2` (`nodeid`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 AUTO_INCREMENT=9 ;

--
-- Dumping data for table `tbchannel`
--

INSERT INTO `tbchannel` (`id`, `nodeid`, `nodename`, `user`, `channame`, `chantype`, `content`, `ip`, `usertime_dt`, `create_dt`) VALUES
(1, '1', 'andromeda', 'dangermouse', 'Altos', 'CHAT', 'Hello World', '127.0.0.1', '2013-02-18 00:00:00', '2013-02-18 00:00:00'),
(2, '2', 'virgibuntu', 'virus', 'Altos', 'CHAT', 'hey Tiz', NULL, NULL, '0000-00-00 00:00:00'),
(3, '1', 'andromeda', 'dangermouse', 'Plaza', 'CHAT', 'another chat entry in another channel', NULL, NULL, '0000-00-00 00:00:00'),
(4, '2', 'virgibuntu', 'virginia', 'Altos', 'CHAT', 'test', '127.0.0.1', NULL, '2013-02-19 16:26:15'),
(6, '2', 'virgibuntu', 'virginia', 'Altos', 'CHAT', 'test', '127.0.0.1', NULL, '2013-02-21 11:37:59'),
(7, '2', 'virgibuntu', 'virginia', 'Altos', 'CHAT', 'test', '127.0.0.1', NULL, '2013-02-21 11:38:53'),
(8, '2', 'virgibuntu', 'virginia', 'Altos', 'CHAT', 'test', '127.0.0.1', NULL, '2013-02-21 11:38:56');

-- --------------------------------------------------------

--
-- Table structure for table `tbclient`
--

CREATE TABLE IF NOT EXISTS `tbclient` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `nodeid` varchar(42) NOT NULL,
  `nodename` varchar(32) NOT NULL,
  `country` varchar(32) NOT NULL,
  `region` varchar(32) DEFAULT NULL,
  `city` varchar(32) DEFAULT NULL,
  `zip` varchar(32) DEFAULT NULL,
  `ip` varchar(32) DEFAULT NULL,
  `port` varchar(32) DEFAULT NULL,
  `localip` varchar(32) DEFAULT NULL,
  `os` varchar(32) NOT NULL,
  `version` double NOT NULL,
  `acceptincoming` int(11) NOT NULL DEFAULT '0',
  `gigaflops` int(11) NOT NULL,
  `ram` int(11) NOT NULL,
  `mhz` int(11) NOT NULL,
  `nbcpus` int(11) NOT NULL,
  `bits` int(11) NOT NULL,
  `isscreensaver` int(11) NOT NULL DEFAULT '0',
  `uptime` double NOT NULL,
  `totaluptime` double NOT NULL,
  `longitude` double NOT NULL,
  `latitude` double NOT NULL,
  `userid` varchar(32) NOT NULL,
  `team` varchar(64) NOT NULL,
  `description` varchar(256) DEFAULT NULL,
  `cputype` varchar(64) DEFAULT NULL,
  `create_dt` datetime NOT NULL,
  `update_dt` datetime DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `nodeid` (`nodeid`),
  KEY `nodeid_2` (`nodeid`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 AUTO_INCREMENT=6 ;

--
-- Dumping data for table `tbclient`
--

INSERT INTO `tbclient` (`id`, `nodeid`, `nodename`, `country`, `region`, `city`, `zip`, `ip`, `port`, `localip`, `os`, `version`, `acceptincoming`, `gigaflops`, `ram`, `mhz`, `nbcpus`, `bits`, `isscreensaver`, `uptime`, `totaluptime`, `longitude`, `latitude`, `userid`, `team`, `description`, `cputype`, `create_dt`, `update_dt`) VALUES
(1, '1', 'andromeda', 'Switzerland', NULL, NULL, NULL, NULL, NULL, NULL, 'Win7', 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 46.5, '', '', NULL, NULL, '0000-00-00 00:00:00', NULL),
(2, '2', 'virgibuntu', 'Switzerland', NULL, NULL, NULL, NULL, NULL, NULL, 'WinXP', 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 47, '', '', NULL, NULL, '0000-00-00 00:00:00', NULL),
(5, '4', 'blabla', '', '', '', '', '127.0.0.1', '', '', '', 0, 0, 0, 0, 0, 0, 32, 0, 0, 9, 0, 0, '', '', '', '', '2013-02-25 15:57:26', '2013-02-26 14:43:24');

-- --------------------------------------------------------

--
-- Table structure for table `tbjobdefinition`
--

CREATE TABLE IF NOT EXISTS `tbjobdefinition` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `jobdefinitionid` varchar(42) NOT NULL,
  `job` varchar(1024) NOT NULL,
  `nodename` varchar(32) NOT NULL,
  `nodeid` varchar(42) NOT NULL,
  `ip` varchar(32) DEFAULT NULL,
  `create_dt` datetime NOT NULL,
  `update_dt` datetime DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `jobdefinitionid` (`jobdefinitionid`),
  KEY `jobdefinitionid_2` (`jobdefinitionid`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 AUTO_INCREMENT=9 ;

--
-- Dumping data for table `tbjobdefinition`
--

INSERT INTO `tbjobdefinition` (`id`, `jobdefinitionid`, `job`, `nodename`, `nodeid`, `ip`, `create_dt`, `update_dt`) VALUES
(1, 'ac43b', '1,1,add', 'andromeda', '1', NULL, '2013-02-27 00:00:00', '2013-02-27 00:00:00'),
(2, 'ac44b', '3,2,add', 'andromeda', '1', '127.0.0.1', '2013-02-27 16:06:47', '2013-02-27 16:06:47'),
(3, 'ac44bd', '4,2,add', 'andromeda', '1', '127.0.0.1', '2013-02-27 17:52:09', '2013-02-27 17:52:09'),
(4, 'ac44dbd', '6,2,add', 'andromeda', '1', '127.0.0.1', '2013-02-27 17:52:36', '2013-02-27 17:52:36'),
(5, 'ac44dbde', '6,2,add', 'andromeda', '1', '127.0.0.1', '2013-02-27 17:53:58', '2013-02-27 17:53:58'),
(6, 'ac43dbde', '6,2,add', 'andromeda', '1', '127.0.0.1', '2013-02-27 17:56:01', '2013-02-27 17:56:01'),
(7, 'ac43dbd2e', '6,2,add', 'andromeda', '1', '127.0.0.1', '2013-02-27 17:57:06', '2013-02-27 17:57:06'),
(8, 'ac43ddbd2e', '7,2,add', 'andromeda', '1', '127.0.0.1', '2013-02-27 17:57:55', '2013-02-27 17:57:55');

-- --------------------------------------------------------

--
-- Table structure for table `tbjobqueue`
--

CREATE TABLE IF NOT EXISTS `tbjobqueue` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `jobdefinitionid` varchar(42) NOT NULL,
  `jobqueueid` varchar(42) NOT NULL,
  `workunitjob` varchar(64) DEFAULT NULL,
  `workunitresult` varchar(64) DEFAULT NULL,
  `nodeid` varchar(42) NOT NULL,
  `nodename` varchar(32) DEFAULT NULL,
  `create_dt` datetime NOT NULL,
  `transmission_dt` datetime DEFAULT NULL,
  `reception_dt` datetime DEFAULT NULL,
  `ip` varchar(42) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `jobqueueid` (`jobqueueid`),
  KEY `jobqueueid_2` (`jobqueueid`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 AUTO_INCREMENT=19 ;

--
-- Dumping data for table `tbjobqueue`
--

INSERT INTO `tbjobqueue` (`id`, `jobdefinitionid`, `jobqueueid`, `workunitjob`, `workunitresult`, `nodeid`, `nodename`, `create_dt`, `transmission_dt`, `reception_dt`, `ip`) VALUES
(1, 'ac43b', 'jqid', 'workunitjob', 'workunitresult', '1', NULL, '2013-02-06 00:00:00', NULL, NULL, ''),
(2, 'ac43dbd2e', '8b3bb012115428c0cb07745357c28822', '', '', '1', 'andromeda', '2013-02-27 17:57:06', NULL, NULL, '127.0.0.1'),
(3, 'ac43ddbd2e', '3bf69edcd1c15bab52947f235c09a44b', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(4, 'ac43ddbd2e', 'f2b6f4e2d72d0cdf2d5c7c10fb0eab32', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(5, 'ac43ddbd2e', '3eb102a527a7e4d720a196694947f5a7', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(6, 'ac43ddbd2e', '5216378825321f2498747908293a3579', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(7, 'ac43ddbd2e', '7c89182ed3a75ff45254ed46fb1e9b10', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(8, 'ac43ddbd2e', '899edba8c996f2921fa742db8635163d', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(9, 'ac43ddbd2e', '791919b4d542d693ebcf40cfc3d2719c', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(10, 'ac43ddbd2e', 'f85811eacc3a7447134a434e023350a5', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(11, 'ac43ddbd2e', '027d5d503db94354d6464e38bc08303e', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(12, 'ac43ddbd2e', 'df4683f5711fbc12846ecadc739dd728', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(13, 'ac43ddbd2e', '0290133976f6892e270a4cc049e2d477', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(14, 'ac43ddbd2e', '3eeca23f7f5a675507561db5000d1576', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(15, 'ac43ddbd2e', '656388d66d725b56b2c5c5033ccfa3d0', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(16, 'ac43ddbd2e', '3ca7f305050132c88333bf7377aa2438', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(17, 'ac43ddbd2e', '26b9b205a59f6933472cd3ef462609d8', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1'),
(18, 'ac43ddbd2e', '545836148e43b2335927328c29fac323', '', '', '1', 'andromeda', '2013-02-27 17:57:55', NULL, NULL, '127.0.0.1');

-- --------------------------------------------------------

--
-- Table structure for table `tbjobresult`
--

CREATE TABLE IF NOT EXISTS `tbjobresult` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `jobresultid` varchar(42) NOT NULL,
  `jobid` varchar(42) NOT NULL,
  `jobqueueid` varchar(42) NOT NULL,
  `jobresult` varchar(1024) NOT NULL,
  `workunitresult` varchar(64) NOT NULL,
  `iserroneous` int(11) NOT NULL DEFAULT '0',
  `errorid` int(11) NOT NULL DEFAULT '0',
  `errorarg` varchar(32) NOT NULL,
  `errormsg` varchar(32) NOT NULL,
  `nodename` varchar(32) NOT NULL,
  `nodeid` varchar(42) NOT NULL,
  `ip` varchar(32) DEFAULT NULL,
  `create_dt` datetime NOT NULL,
  PRIMARY KEY (`id`),
  KEY `jobresultid` (`jobresultid`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 AUTO_INCREMENT=2 ;

--
-- Dumping data for table `tbjobresult`
--

INSERT INTO `tbjobresult` (`id`, `jobresultid`, `jobid`, `jobqueueid`, `jobresult`, `workunitresult`, `iserroneous`, `errorid`, `errorarg`, `errormsg`, `nodename`, `nodeid`, `ip`, `create_dt`) VALUES
(1, 'jrid', 'ac43b', 'jqid', '2', 'workunitresult', 0, 0, '', '', 'andromeda', '1', NULL, '2013-02-27 00:00:00');

-- --------------------------------------------------------

--
-- Table structure for table `tbparameter`
--

CREATE TABLE IF NOT EXISTS `tbparameter` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `paramtype` varchar(20) COLLATE latin1_general_ci NOT NULL,
  `paramname` varchar(32) COLLATE latin1_general_ci NOT NULL,
  `paramvalue` varchar(255) COLLATE latin1_general_ci NOT NULL,
  `create_dt` datetime DEFAULT NULL,
  `update_dt` datetime DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `paramtype` (`paramtype`,`paramname`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 COLLATE=latin1_general_ci AUTO_INCREMENT=14 ;

--
-- Dumping data for table `tbparameter`
--

INSERT INTO `tbparameter` (`id`, `paramtype`, `paramname`, `paramvalue`, `create_dt`, `update_dt`) VALUES
(1, 'TEST', 'DB_CONNECTION', 'OK', NULL, '2013-02-25 16:13:37'),
(9, 'TIME', 'UPTIME', '80989', '2013-02-25 11:55:44', '2013-02-26 14:43:27'),
(8, 'CONFIGURATION', 'SERVER_ID', 'fb4bc9a27a2be5e0b7ce08dc2bf09618', '2013-02-25 11:55:44', '2013-02-25 11:55:44'),
(11, 'SECURITY', 'PWD_HASH_SALT', 'caacafd10c3a5837a9f98e21991e4d22', '2013-02-25 11:55:44', '2013-02-25 11:55:44'),
(13, 'TIME', 'LAST_SUPERSERVER_CALL', '1361886206', '2013-02-25 16:13:37', '2013-02-26 14:43:27');

-- --------------------------------------------------------

--
-- Table structure for table `tbserver`
--

CREATE TABLE IF NOT EXISTS `tbserver` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `serverid` varchar(42) NOT NULL,
  `servername` varchar(32) NOT NULL,
  `serverurl` varchar(256) NOT NULL,
  `chatchannel` varchar(32) DEFAULT NULL,
  `version` double NOT NULL,
  `superserver` tinyint(1) NOT NULL,
  `ip` varchar(32) DEFAULT NULL,
  `uptime` double NOT NULL,
  `longitude` double NOT NULL,
  `latitude` double NOT NULL,
  `activenodes` int(11) NOT NULL,
  `jobinqueue` int(11) NOT NULL,
  `create_dt` datetime NOT NULL,
  `update_dt` datetime DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `serverid` (`serverid`),
  KEY `serverid_2` (`serverid`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=13 ;

--
-- Dumping data for table `tbserver`
--

INSERT INTO `tbserver` (`id`, `serverid`, `servername`, `serverurl`, `chatchannel`, `version`, `superserver`, `ip`, `uptime`, `longitude`, `latitude`, `activenodes`, `jobinqueue`, `create_dt`, `update_dt`) VALUES
(9, 'fb4bc9a27a2be5e0b7ce08dc2bf09618', 'Altos', '127.0.0.1:8090/gpu_freedom/src/server', 'altos', 0.1, 0, 'localhost', 80989, 14, 10, 3, 0, '2013-02-25 16:27:29', '2013-02-26 14:43:30'),
(11, '6e771f4936a0d24bf2448e0d187725a4', 'Orion', '127.0.0.1:8090/server', 'orion', 0.1, 1, '', 1693, 14, 10, 0, 0, '2013-02-26 14:35:36', '2013-02-27 08:40:01'),
(12, 'paripara', 'Algol', 'http://127.0.0.1:8090/algol', 'algol', 0.05, 0, '', 99, 90, 90, 13, 2, '2013-02-26 14:39:33', '2013-02-27 08:40:02');

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
