-- phpMyAdmin SQL Dump
-- version 3.5.2
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Feb 21, 2013 at 03:05 PM
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
  PRIMARY KEY (`id`)
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
  PRIMARY KEY (`id`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 AUTO_INCREMENT=5 ;

--
-- Dumping data for table `tbclient`
--

INSERT INTO `tbclient` (`id`, `nodeid`, `nodename`, `country`, `region`, `city`, `zip`, `ip`, `port`, `localip`, `os`, `version`, `acceptincoming`, `gigaflops`, `ram`, `mhz`, `nbcpus`, `bits`, `isscreensaver`, `uptime`, `totaluptime`, `longitude`, `latitude`, `userid`, `team`, `description`, `cputype`, `create_dt`, `update_dt`) VALUES
(1, '1', 'andromeda', 'Switzerland', NULL, NULL, NULL, NULL, NULL, NULL, 'Win7', 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 46.5, '', '', NULL, NULL, '0000-00-00 00:00:00', NULL),
(2, '2', 'virgibuntu', 'Switzerland', NULL, NULL, NULL, NULL, NULL, NULL, 'WinXP', 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 47, '', '', NULL, NULL, '0000-00-00 00:00:00', NULL),
(3, '4', 'piero', '', '', '', '', '127.0.0.1', '', '', '', 0, 0, 0, 0, 0, 0, 32, 0, 0, 11, 0, 0, '', '', '', '', '2013-02-21 11:58:24', '2013-02-21 14:59:15'),
(4, '4', 'blabla', '', '', '', '', '127.0.0.1', '', '', '', 0, 0, 0, 0, 0, 0, 32, 0, 0, 9, 0, 0, '', '', '', '', '2013-02-21 11:58:49', '2013-02-21 11:58:49');

-- --------------------------------------------------------

--
-- Table structure for table `tbjob`
--

CREATE TABLE IF NOT EXISTS `tbjob` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `jobid` varchar(16) NOT NULL,
  `job` varchar(1024) NOT NULL,
  `workunitincoming` varchar(64) NOT NULL,
  `workunitoutgoing` varchar(64) NOT NULL,
  `requests` int(11) NOT NULL DEFAULT '1',
  `delivered` int(11) NOT NULL DEFAULT '0',
  `results` int(11) NOT NULL DEFAULT '0',
  `nodename` varchar(64) NOT NULL,
  `nodeid` varchar(42) NOT NULL,
  `ip` varchar(32) DEFAULT NULL,
  `create_dt` datetime NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1 AUTO_INCREMENT=1 ;

-- --------------------------------------------------------

--
-- Table structure for table `tbjobqueue`
--

CREATE TABLE IF NOT EXISTS `tbjobqueue` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `job_id` int(11) NOT NULL,
  `nodeid` varchar(42) NOT NULL,
  `transmitted` int(11) NOT NULL DEFAULT '0',
  `received` int(11) NOT NULL DEFAULT '0',
  `create_dt` datetime NOT NULL,
  `transmission_dt` datetime DEFAULT NULL,
  `reception_dt` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1 AUTO_INCREMENT=1 ;

-- --------------------------------------------------------

--
-- Table structure for table `tbjobresult`
--

CREATE TABLE IF NOT EXISTS `tbjobresult` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `job_id` int(11) NOT NULL,
  `jobid` varchar(16) NOT NULL,
  `jobqueue_id` int(11) NOT NULL,
  `jobresult` varchar(1024) NOT NULL,
  `workunitresult` varchar(64) NOT NULL,
  `iserroneous` int(11) NOT NULL DEFAULT '0',
  `errorid` int(11) NOT NULL DEFAULT '0',
  `errorarg` varchar(32) NOT NULL,
  `errormsg` varchar(32) NOT NULL,
  `nodename` varchar(64) NOT NULL,
  `nodeid` varchar(42) NOT NULL,
  `ip` varchar(32) DEFAULT NULL,
  `create_dt` datetime NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1 AUTO_INCREMENT=1 ;

-- --------------------------------------------------------

--
-- Table structure for table `tbparameter`
--

CREATE TABLE IF NOT EXISTS `tbparameter` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `paramtype` varchar(20) COLLATE latin1_general_ci NOT NULL,
  `paramname` varchar(20) COLLATE latin1_general_ci NOT NULL,
  `paramvalue` varchar(255) COLLATE latin1_general_ci NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `paramtype` (`paramtype`,`paramname`)
) ENGINE=MyISAM  DEFAULT CHARSET=latin1 COLLATE=latin1_general_ci AUTO_INCREMENT=3 ;

--
-- Dumping data for table `tbparameter`
--

INSERT INTO `tbparameter` (`id`, `paramtype`, `paramname`, `paramvalue`) VALUES
(1, 'TEST', 'DB_CONNECTION', 'OK');

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
  `totaluptime` double NOT NULL,
  `longitude` double NOT NULL,
  `latitude` double NOT NULL,
  `activenodes` int(11) NOT NULL,
  `jobinqueue` int(11) NOT NULL,
  `create_dt` datetime NOT NULL,
  `update_dt` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=3 ;

--
-- Dumping data for table `tbserver`
--

INSERT INTO `tbserver` (`id`, `serverid`, `servername`, `serverurl`, `chatchannel`, `version`, `superserver`, `ip`, `uptime`, `totaluptime`, `longitude`, `latitude`, `activenodes`, `jobinqueue`, `create_dt`, `update_dt`) VALUES
(1, '1', 'Altos', 'http://127.0.0.1:8090/gpu_freedom/src/server', 'altos', 0, 0, '127.0.0.1', 0, 0, 47, 7, 3, 1, '0000-00-00 00:00:00', NULL),
(2, '2', 'Orion', 'http://127.0.0.1:8090/superserver', 'orion', 0, 1, '127.0.0.1', 0, 0, 30, 3, 12, 3, '0000-00-00 00:00:00', NULL);

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
